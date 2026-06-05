
import os
import shutil

import torch
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets.folder import default_loader


class CIFARDataset(object):
    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset

    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean=[0.507, 0.487, 0.441]
        std=[0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

class SVHNDataset(object):
    @staticmethod
    def get_svhn_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform)
        return trainset

    @staticmethod
    def get_svhn_test(path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        return testset

class CINIC10Dataset(object):
    @staticmethod
    def get_cinic10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_cinic10_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        path = os.path.join(path, 'test')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

class ImageNetDataset(object):
    @staticmethod
    def get_ImageNet_train(path, transform=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]))


        return trainset

    @staticmethod
    def get_ImageNet_test(path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        testset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]))
        return testset


class SplitImageDataset(torch.utils.data.Dataset):
    """
    Image-classification dataset defined by a text file that lists image paths
    relative to `root`. The class label of each image is derived from the
    top-level folder in its path, using a deterministic (sorted) class -> index
    mapping that matches torchvision.datasets.ImageFolder ordering.
    """
    def __init__(self, root, split_file, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader

        # Deterministic class ordering, identical to ImageFolder.
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        with open(split_file) as f:
            rel_paths = [line.strip().replace('\\', '/') for line in f if line.strip()]

        self.samples = []
        for rel in rel_paths:
            cls = rel.split('/')[0]
            self.samples.append((os.path.join(root, *rel.split('/')), self.class_to_idx[cls]))
        self.targets = [target for _, target in self.samples]

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class MITIndoor67Dataset(object):
    """
    MIT Indoor Scene Recognition dataset: 67 indoor categories, ~15620 images.
    See https://web.mit.edu/torralba/www/indoor.html

    Expected directory layout (as distributed):
        <root>/Images/<category>/<category>_<id>.jpg
        <root>/TrainImages.txt   (5360 images, 80 per class)
        <root>/TestImages.txt    (1340 images, 20 per class)
    Each line of the split files is a path relative to the images dir, e.g.
        airport_inside/airport_inside_0001.jpg
    If there is no `Images/` subfolder, the category folders are assumed to live
    directly under <root>.
    """

    num_classes = 67

    # ImageNet statistics — the standard choice for training/fine-tuning on MIT67.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    @staticmethod
    def _images_dir(path):
        images_dir = os.path.join(path, 'Images')
        return images_dir if os.path.isdir(images_dir) else path

    @staticmethod
    def get_MITIndoor67_train(path, transform=None):
        if transform is None:
            normalize = transforms.Normalize(mean=MITIndoor67Dataset.mean,
                                             std=MITIndoor67Dataset.std)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        return SplitImageDataset(
            root=MITIndoor67Dataset._images_dir(path),
            split_file=os.path.join(path, 'TrainImages.txt'),
            transform=transform)

    @staticmethod
    def get_MITIndoor67_test(path, transform=None):
        if transform is None:
            normalize = transforms.Normalize(mean=MITIndoor67Dataset.mean,
                                             std=MITIndoor67Dataset.std)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return SplitImageDataset(
            root=MITIndoor67Dataset._images_dir(path),
            split_file=os.path.join(path, 'TestImages.txt'),
            transform=transform)