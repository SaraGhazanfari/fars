import os
from os.path import join, exists

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DataAugmentationDINO:

    def __init__(self, global_crops_scale=None, local_crops_scale=None, local_crops_number=None):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.standard_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.CenterCrop(224),
            flip_and_color_jitter,
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.standard_transform(image)
        # images = []
        # images.append(self.standard_transform(image))
        # images.append(self.global_transfo1(image))
        # images = torch.stack(images, dim=0)
        # return images


class BaseReader:

    def __init__(self, config, batch_size, is_distributed, is_training):
        self.config = config
        self.batch_size = batch_size
        self.is_training = is_training
        self.is_distributed = is_distributed
        self.num_workers = 1
        self.prefetch_factor = self.batch_size * 2
        self.path = join(self.get_data_dir(), self.config.dataset)

    def get_data_dir(self):
        paths = self.config.data_dir.split(':')
        data_dir = None
        for path in paths:
            if exists(join(path, self.config.dataset)):
                data_dir = path
                break
        if data_dir is None:
            raise ValueError("Data directory not found.")
        return data_dir

    def transform(self):
        """Create the transformer pipeline."""
        raise NotImplementedError('Must be implemented in derived classes')

    def load_dataset(self):
        """Load or download dataset."""
        sampler = None

        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=self.is_training and not sampler,
                            pin_memory=False,
                            prefetch_factor=self.prefetch_factor,
                            sampler=sampler)
        return loader, sampler


class ImagenetReader(BaseReader):

    def __init__(self, config, batch_size, is_training, is_distributed=False):
        super(ImagenetReader, self).__init__(
            config, batch_size, is_distributed, is_training)
        self.config = config
        self.batch_size = batch_size
        self.is_training = is_training
        self.n_classes = 1000
        self.height, self.width = 224, 500
        self.n_train_files = 1_281_167
        self.n_test_files = 50_000
        self.img_size = (None, 3, 224, 500)
        self.split = 'train' if self.is_training else 'val'
        # todo put a better value for imagenet, it is basically like a bias
        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)

        if is_training:
            transform = DataAugmentationDINO(
                global_crops_scale=(0.4, 1.),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=8
            )

        else:
            transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        split = 'train' if is_training else 'val'
        self.dataset = ImageFolder(os.path.join(self.path, split), transform=transform)


class Imagenet100Reader(ImagenetReader):
    def __init__(self, config, batch_size, is_training, is_distributed=False):
        super(Imagenet100Reader, self).__init__(
            config, batch_size, is_distributed, is_training)
        self.n_train_files = 130_301
        self.n_test_files = 5_301
        self.n_classes = 100


class Imagenet10Reader(ImagenetReader):
    def __init__(self, config, batch_size, is_training, is_distributed=False):
        super(Imagenet10Reader, self).__init__(
            config, batch_size, is_distributed, is_training)
        self.n_train_files = 9469
        self.n_test_files = 3925
        self.n_classes = 10


readers_config = {
    'imagenet': ImagenetReader,
    'imagenet100': Imagenet100Reader,
    'imagenette': Imagenet10Reader,
}
