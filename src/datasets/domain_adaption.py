"""Datasets for domain adaption."""

import os
import torch.utils.data
import torch.multiprocessing as multiprocessing
import torchvision.datasets
import torchvision.transforms as transforms
import itertools

import utils
from datasets import BaseDataset, DATA_ROOT


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def office_amazon_dataset(batch_size, shuffle):
    """Create a dataset from the amazon part of Office-31."""
    trans = transforms.Compose([transforms.CenterCrop(244),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    data_path = os.path.join(DATA_ROOT, 'Office-31', 'amazon', 'images')
    data = torchvision.datasets.ImageFolder(data_path, transform=trans)

    return BaseDataset(data, data, batch_size, shuffle, dvc_file=os.path.join(DATA_ROOT, 'Office-31.dvc'))


def office_webcam_dataset(batch_size, shuffle):
    """Create a dataset from the webcam part of Office-31."""
    trans = transforms.Compose([transforms.CenterCrop(244),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    data_path = os.path.join(DATA_ROOT, 'Office-31', 'webcam', 'images')
    data = torchvision.datasets.ImageFolder(data_path, transform=trans)

    return BaseDataset(data, data, batch_size, shuffle, dvc_file=os.path.join(DATA_ROOT, 'Office-31.dvc'))


def office_dslr_dataset(batch_size, shuffle):
    """Create a dataset from the dslr part of Office-31."""
    trans = transforms.Compose([transforms.CenterCrop(244),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    data_path = os.path.join(DATA_ROOT, 'Office-31', 'dslr', 'images')
    data = torchvision.datasets.ImageFolder(data_path, transform=trans)

    return BaseDataset(data, data, batch_size, shuffle, dvc_file=os.path.join(DATA_ROOT, 'Office-31.dvc'))


def _build_datasets(source_dataset, target_dataset, kwargs):
    """Build the source and target datasets from config dicts."""
    source_dataset['parameters'] = {**source_dataset['parameters'], **kwargs}
    source_config = source_dataset
    source_dataset = utils.build_object(source_config)

    target_dataset['parameters'] = {**target_dataset['parameters'], **kwargs}
    target_config = target_dataset
    target_dataset = utils.build_object(target_config)

    return source_dataset, target_dataset


def deep_adaption_dataset(source_dataset, target_dataset, **kwargs):
    """Build a domain adaption dataset from two datasets."""
    if isinstance(source_dataset, dict) and isinstance(target_dataset, dict):
        source_dataset, target_dataset = _build_datasets(source_dataset, target_dataset, kwargs)

    return DADataset((source_dataset.train_data, target_dataset.train_data),
                     (source_dataset.eval_data, target_dataset.eval_data),
                     dvc_file=(source_dataset.dvc_file, target_dataset.dvc_file),
                     **kwargs)


def no_adaption_dataset(source_dataset, target_dataset, **kwargs):
    """Build a dataset with training data from source and eval data from target dataset."""
    if isinstance(source_dataset, dict) and isinstance(target_dataset, dict):
        source_dataset, target_dataset = _build_datasets(source_dataset, target_dataset, kwargs)

    return BaseDataset(source_dataset.train_data,
                       target_dataset.eval_data,
                       dvc_file=(source_dataset.dvc_file, target_dataset.dvc_file),
                       **kwargs)


class DADataset(BaseDataset):
    """Domain adaption dataset to load data from two datasets concurrently."""

    def _build_data_loader(self, data, batch_size, shuffle, num_workers):
        """Return a ConcatDataLoader that loads source and target data concurrently."""
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        loader = ConcatDataLoader(data[0], data[1],
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)

        return loader


class ConcatDataLoader(torch.utils.data.DataLoader):
    """Dataloader for loading two datasets concurrently."""

    def __init__(self, source, target, **kwargs):
        """
        Create a new dataloader that zips data from two datasets.

        This dataloader takes two datasets to iterate over them
        as if using zip. The shorter of the two datasets will loop
        around to make it as long as the other dataset.

        :param source: source dataset
        :param target: target dataset
        :param kwargs: kwargs to be passed to the dataloader
        """
        # Lengthen the shorter of the two datasets
        if len(source) > len(target):
            times = len(source) // len(target) + 1
            target = torch.utils.data.ConcatDataset([target] * times)
            target = torch.utils.data.Subset(target, range(len(source)))
        else:
            times = len(target) // len(source) + 1
            source = torch.utils.data.ConcatDataset([source] * times)
            source = torch.utils.data.Subset(source, range(len(target)))

        # Initialize super as primary dataloader with one dataset
        super(ConcatDataLoader, self).__init__(source, **kwargs)

        # Initialize secondary dataloader with other dataset
        self.target_loader = torch.utils.data.DataLoader(target, **kwargs)

    def __iter__(self):
        """Return an iterator over the zipped version of the datasets."""
        combined_data_loader = zip(super(ConcatDataLoader, self).__iter__(), self.target_loader)
        combined_data_loader = map(itertools.chain.from_iterable, combined_data_loader)

        return combined_data_loader

    def __len__(self):
        return min(super(ConcatDataLoader, self).__len__(), len(self.target_loader))
