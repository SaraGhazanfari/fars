import glob
from os.path import join, exists

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import DataParallel
from tqdm import tqdm

from fars.core import utils
from fars.core.data.readers import readers_config
from fars.core.models.l2_lip.model import L2LipschitzNetwork, NormalizedModel
from fars.core.models.non_lip.model import LinearClassifier


class LinearEvaluation:
    def __init__(self, config):
        self.config = config
        self.train_dir = self.config.train_dir
        self.has_training = True

    def _init_class_properties(self):
        self.is_master = True
        self.embed_dim = 1792

        means = (0.0000, 0.0000, 0.0000)
        stds = (1.0000, 1.0000, 1.0000)

        Reader = readers_config[self.config.dataset]
        self.reader = Reader(config=self.config, batch_size=self.config.batch_size,
                             is_training=False, is_distributed=False)
        self.val_loader, self.val_sampler = self.reader.load_dataset()

        self.train_loader, self.train_sampler = Reader(config=self.config, batch_size=self.config.batch_size,
                                                       is_training=True, is_distributed=False).load_dataset()

        model = L2LipschitzNetwork(self.config, self.embed_dim)
        self.model = NormalizedModel(model, means, stds)
        self.model = self.model.cuda()
        # utils.setup_distributed_training(self.world_size, self.rank)
        self.model = DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.model = self.load_ckpt()
        self.model = self.model.eval()

        self.linear_classifier = LinearClassifier(dim=768, num_labels=self.reader.n_classes,
                                                  num_layers=self.config.num_linear)
        self.linear_classifier = DataParallel(self.linear_classifier, device_ids=range(torch.cuda.device_count()))
        self.linear_classifier = self.linear_classifier.cuda()
        print('Linear model built.')
        self.load_classifier()

        self.linear_classifier = DataParallel(
            self.linear_classifier, device_ids=range(torch.cuda.device_count()))

        self.optimizer = torch.optim.SGD(
            self.linear_classifier.parameters(),
            0.01,  # self.config.lr * self.config.batch_size,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )

        self.metric_logger = utils.MetricLogger(delimiter="  ")

    def load_classifier(self):
        classifier_checkpoint = glob.glob(join(self.config.train_dir, 'checkpoints', 'classifier-*.pth'))
        modified_state_dict = dict()
        if len(classifier_checkpoint) != 0:
            state_dict = torch.load(classifier_checkpoint[-1])['model_state_dict']
            for key, value in state_dict.items():
                modified_state_dict[key.replace('module.', '')] = value
            msg = self.linear_classifier.load_state_dict(modified_state_dict, strict=False)
            self.has_training = False
            print(f'Linear classifier is loaded! {msg}')

    def load_ckpt(self):
        checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
        ckpt_path = join(self.config.train_dir, 'checkpoints', ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)
        return self.model

    def _save_ckpt(self, step, epoch, final=False, best=False):
        """Save ckpt in train directory."""
        freq_ckpt_epochs = self.config.save_checkpoint_epochs
        if (epoch % freq_ckpt_epochs == 0 and self.is_master and epoch not in self.saved_ckpts) or (
                final and self.is_master) or best:
            prefix = "model" if not best else "best_model"
            ckpt_name = f"{prefix}.ckpt-{step}.pth"
            ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
            if exists(ckpt_path) and not best:
                return
            self.saved_ckpts.add(epoch)
            state = {
                'epoch': epoch,
                'global_step': step,
                'model_state_dict': self.linear_classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict()
            }
            torch.save(state, ckpt_path)

    # @record
    def __call__(self):
        self._init_class_properties()
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(self.config)).items())))
        self.evaluate()
        if self.has_training:
            cudnn.benchmark = True
            self.saved_ckpts = set([0])
            best_acc = 0
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.epochs, eta_min=0)

            for epoch in range(0, self.config.epochs):
                self.train(epoch)
                scheduler.step()
                self.evaluate()
                self._save_ckpt(step=epoch, epoch=epoch)
            print("Training of the supervised linear classifier on frozen features completed.\n"
                  "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
            self._save_ckpt(step=self.config.epochs, epoch=self.config.epochs, final=True)

    def train(self, epoch):

        self.linear_classifier.train()
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        for idx, (inp, target) in tqdm(enumerate(self.metric_logger.log_every(self.train_loader, 1, header))):

            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.model(inp)[:, :768]

            output = self.linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(output, target)

            # compute the gradients
            self.optimizer.zero_grad()
            loss.backward()
            if idx % 100 == 99:
                print(
                    f'Epoch: {epoch + 1}, iteration: {idx + 1}/{len(self.train_loader)}, Loss: {round(loss.item(), 4)}')
            # step
            self.optimizer.step()

            # log
            torch.cuda.synchronize()
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def evaluate(self):
        self.linear_classifier.eval()

        for inp, target in self.val_loader:
            # self.val_sampler.set_epoch(0)
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.model(inp)[:, :768]
            output = self.linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            # if self.linear_classifier.module.num_labels >= 5:
            #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # else:
            acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # if self.linear_classifier.module.num_labels >= 5:
            #     self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # if self.linear_classifier.module.num_labels >= 5:
        #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #           .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5, losses=self.metric_logger.loss))
        # else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=self.metric_logger.acc1, losses=self.metric_logger.loss))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
