import datetime
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

N_CLASSES = {
    'dino_vitb16': 768,
    'open_clip_vitb32': 512,
    'clip_vitb32': 512,
    'ensemble': 1792,
    'dinov2_vits14_reg': 768,
    'dinov2_vitb14_reg': 768,
    'dinov2_vits14': 768,
    'dinov2_vitb14': 768,
    'dino_vitb8': 768
}


def get_parameter_number(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_epochs_from_ckpt(filename):
    regex = "(?<=ckpt-)[0-9]+"
    return int(re.findall(regex, filename)[-1])


def setup_logging(config, rank):
    level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
             'INFO': 20, 'WARN': 30
             }[config.logging_verbosity]
    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    # format_ = "[%(asctime)s %(pathname)s:%(lineno)s] %(message)s"
    filename = '{}/log_{}_{}.logs'.format(config.train_dir, config.mode, rank)
    f = open(filename, "a")
    logging.basicConfig(filename=filename, level=level, format=format_, datefmt='%H:%M:%S')


class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, epoch_id):
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return 2 - 2 * torch.mean(cos_sim(x, y))
        # x = F.normalize(x, dim=-1, p=2)
        # y = F.normalize(y, dim=-1, p=2)
        # return 2 - 2 * (x * y).sum(dim=-1)


class RMSELoss(nn.Module):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y, epoch=0):
        return torch.sqrt(self.mse(yhat, y))


class HingeLoss(torch.nn.Module):

    def __init__(self, device, margin):
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x, y):
        y_rounded = torch.round(y)  # Map [0, 1] -> {0, 1}
        y_transformed = -1 * (1 - 2 * y_rounded)  # Map {0, 1} -> {-1, 1}
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()


def get_loss(config, margin=0, device='cuda:0'):
    if config.loss == 'rmse':
        return RMSELoss()
    elif config.loss == 'hinge':
        return HingeLoss(margin=config.margin, device=device)
    elif config.loss == 'byol':
        return BYOLLoss()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        sys.stdout.flush()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def load_pretrained_dino_linear_weights(linear_classifier,
                                        linear_pretrained_weights='checkpoint_Dino-reference_linear-010.pth.tar',
                                        model_name='vit_base', patch_size=16):
    if os.path.isfile(linear_pretrained_weights):
        state_dict = torch.load(linear_pretrained_weights, map_location="cpu")["state_dict"]
        msg = linear_classifier.load_state_dict(state_dict, strict=False)
        print('Linear pretrained weights found at {} and loaded with msg: {}'.format(linear_pretrained_weights, msg))
    else:
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
        if url is not None:
            print("We load the reference pretrained linear weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)[
                "state_dict"]
            linear_classifier.load_state_dict(state_dict, strict=True)
        else:
            print("We use random linear weights.")
