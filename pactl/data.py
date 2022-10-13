import os
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.distributions import Categorical
import torchvision.transforms as transforms
from timm.data import create_dataset, ImageDataset


__all__ = [
    'get_dataset','get_data_dir',
]

_DATASET_CFG = {
    'mnist': {
        'num_classes': 10,
    },
    'fmnist': {
        'num_classes': 10,
    },
    'svhn': {
        'num_classes': 10,
    },
    'svhn28': {
        'num_classes': 10,
    },
    'cifar10': {
        'num_classes': 10,
    },
    'cifar100': {
        'num_classes': 100,
    },
    'tiny-imagenet': {
        'num_classes': 200,
    },
    'imagenet': {
        'num_classes': 1000
    },
}

translation_flip = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])


class WrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
    
    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, __value):
        return setattr(self.dataset, 'targets', __value)

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, __value):
        return setattr(self.dataset, 'transform', __value)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class LabelNoiseDataset(WrapperDataset):
    def __init__(self, dataset, n_labels=10, label_noise=0):
        super().__init__(dataset)

        self.C = n_labels

        if label_noise > 0:
            orig_targets = self.targets
            self.noisy_targets = torch.where(
                torch.rand(len(orig_targets)) < label_noise,
                Categorical(probs=torch.ones(self.C) / self.C).sample(torch.Size([len(orig_targets)])),
                torch.Tensor(orig_targets).long())

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        y = self.noisy_targets[i]
        return X, y


def get_data_dir(data_dir=None):
    if data_dir is None:
        if os.environ.get('DATADIR') is not None:
            data_dir = os.environ.get('DATADIR')
            logging.debug(f'Using default data directory from environment "{data_dir}".')
        else:
            home_data_dir = Path().home() / 'datasets'
            data_dir = str(home_data_dir.resolve())
            logging.debug(f'Using default HOME data directory "{data_dir}".')

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    return data_dir


def get_dataset(dataset, root=None, train_subset=1, label_noise=0,
                indices_path=None, **kwargs):
    root = get_data_dir(data_dir=root)

    if dataset == 'mnist':
        train_data, test_data = get_mnist(root=root,**kwargs)
    elif dataset == 'fmnist':
        train_data, test_data = get_fmnist(root=root,**kwargs)
    elif dataset == 'svhn':
        train_data, test_data = get_svhn(root=root,**kwargs)
    elif dataset == 'svhn28':
        train_data, test_data = get_svhn28(root=root,**kwargs)
    elif dataset == 'cifar10':
        train_data, test_data = get_cifar10(root=root,**kwargs)
    elif dataset == 'cifar100':
        train_data, test_data = get_cifar100(root=root,**kwargs)
    elif dataset == 'tiny-imagenet':
        train_data, test_data = get_tiny_imagenet(root=root,**kwargs)
    elif dataset == 'imagenet':
        train_data, test_data = get_imagenet(root=root,**kwargs)
    else:
        raise NotImplementedError

    num_classes = _DATASET_CFG[dataset]['num_classes']

    if label_noise > 0:
        train_data = LabelNoiseDataset(train_data, n_labels=num_classes, label_noise=label_noise)

    if np.abs(train_subset) < 1:
        n = len(train_data)
        ns = int(n * np.abs(train_subset))

        randperm = np.load(indices_path) if indices_path is not None else torch.randperm(n)
        assert len(randperm) == n, f'Permutation length {len(randperm)} does not match dataset length {n}'

        ## NOTE: -ve train_subset fraction to get latter segment.
        randperm = randperm[ns:] if train_subset < 0 else randperm[:ns]
        train_data = Subset(train_data, randperm)

    logging.info(f'Train Dataset Size: {len(train_data)};  Test Dataset Size: {len(test_data)}')

    setattr(train_data, 'num_classes', num_classes)
    return train_data, test_data


def get_mnist(root=None, extra_transform=None, **_):
    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(create_dataset('torch/mnist', root=root, split=split,
                                transform=transform, download=True))
        datasets[-1].num_inputs = 1
    return datasets


def get_fmnist(root=None, extra_transform=None, **_):
    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(create_dataset('torch/fashion_mnist', root=root, split=split,
                                       transform=transform, download=True))
        datasets[-1].num_inputs = 1
    return datasets


def get_svhn(root=None, extra_transform=None, aug=True):
    '''Dataset SVHN

    root (str): Root directory where 'svhn' folder exists or will be downloaded to.
    '''
    from torchvision.datasets import SVHN

    (Path(root) / 'svhn').mkdir(parents=True, exist_ok=True)

    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            translation_flip if split=='train' and aug else transforms.Compose([]),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(SVHN(root=Path(root) / 'svhn', split=split,
                             transform=transform, download=True))
        datasets[-1].num_inputs = 3
    return datasets


def get_svhn28(root=None, extra_transform=None, aug=True):
    '''Dataset SVHN

    root (str): Root directory where 'svhn' folder exists or will be downloaded to.
    '''
    from torchvision.datasets import SVHN

    (Path(root) / 'svhn').mkdir(parents=True, exist_ok=True)

    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            translation_flip if split=='train' and aug else transforms.Compose([]),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(SVHN(root=Path(root) / 'svhn', split=split,
                             transform=transform, download=True))
        datasets[-1].num_inputs = 3
    return datasets


def get_cifar10(root=None, extra_transform=None, aug=True):
    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            translation_flip if split=='train' and aug else transforms.Compose([]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (.247,.243,.261)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(create_dataset('torch/cifar10', root=root, split=split,
                                transform=transform, download=True))
        datasets[-1].num_inputs = 3
    return datasets


def get_cifar100(root=None, extra_transform=None, aug=True):
    datasets = []
    for split in ['train', 'test']:
        transform = transforms.Compose([
            translation_flip if split=='train' and aug else transforms.Compose([]),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761)),
        ])
        if extra_transform is not None:
            transform = transforms.Compose([ extra_transform, transform ])
        datasets.append(create_dataset('torch/cifar100', root=root, split=split,
                                transform=transform, download=True))
        datasets[-1].num_inputs = 3
    return datasets


def get_tiny_imagenet(root=None):
    _TINY_IMAGENET_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])
    _TINY_IMAGENET_TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    train_data = ImageDataset(root=Path(root) / 'tiny-imagenet-200' / 'train',
                             transform=_TINY_IMAGENET_TRAIN_TRANSFORM)

    val_data = ImageDataset(root=Path(root) / 'tiny-imagenet-200' / 'val',
                            transform=_TINY_IMAGENET_TEST_TRANSFORM)
    
    ## NOTE: Folder not in the right format.
    # test_data = ImageFolder(root=Path(root) / 'tiny-imagenet-200' / 'test', transform=_TINY_IMAGENET_TEST_TRANSFORM)

    return train_data, val_data


def get_imagenet(root=None):
    _IMAGENET_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _IMAGENET_TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_data = ImageDataset(root=Path(root) / 'imagenet' / 'train',
                             transform=_IMAGENET_TRAIN_TRANSFORM)

    val_data = ImageDataset(root=Path(root) / 'imagenet' / 'val',
                            transform=_IMAGENET_TEST_TRANSFORM)
    
    return train_data, val_data
