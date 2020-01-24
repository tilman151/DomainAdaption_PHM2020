"""Dataset abstraction classes."""

import os
import torch.utils.data
import multiprocessing
import mlflow

import utils


DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
os.makedirs(DATA_ROOT, exist_ok=True)


class BaseDataset:
    """Base class for datasets."""

    def __init__(self, train_data, eval_data, batch_size, shuffle=True, dvc_file=None):
        """
        Construct a new dataset.

        This dataset contains the training and evaluation data for a dataset. It can
        return a dataloader for each of them with the predetermined batch size and
        shuffle. The DVC file of the dataset on disk will be logged to mlflow as
        an artiact.

        :param train_data: torch.Dataset of the training data
        :param eval_data: torch.Dataset of the evaluation data
        :param batch_size: default batch size
        :param shuffle: default shuffle
        :param dvc_file: dvc file path of data
        """
        self.train_data = train_data
        self.eval_data = eval_data

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dvc_file = dvc_file
        if self.dvc_file is not None and not utils.is_debugging():
            dvc_file = dvc_file if isinstance(dvc_file, tuple) else (dvc_file,)
            for f in dvc_file:
                mlflow.log_artifact(f, artifact_path='data_version')

    def training(self, batch_size=None, shuffle=None, num_workers=None):
        """Return a dataloader of the training data."""
        if self.train_data is not None:
            return self._build_data_loader(self.train_data, batch_size, shuffle, num_workers)
        else:
            raise ValueError('No training data specified for this dataset.')

    def evaluation(self, batch_size=None, shuffle=None, num_workers=None):
        """Return a dataloader of the evaluation data."""
        if self.eval_data is not None:
            return self._build_data_loader(self.eval_data, batch_size, shuffle, num_workers)
        else:
            raise ValueError('No evaluation data specified for this dataset.')

    def _build_data_loader(self, data, batch_size, shuffle, num_workers):
        """Build a dataloader with default or explicit parameters."""
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size,
                                                  shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
        return data_loader

    def __str__(self):
        """Return a string representation of the dataset."""
        desc = 'BaseDataset(' + os.linesep
        desc += '\tTraining data' + os.linesep
        desc += '\t\t' + str(self.train_data).replace('\n', os.linesep + '\t') + os.linesep
        desc += '\tEvaluation data' + os.linesep
        desc += '\t\t' + str(self.eval_data).replace('\n', os.linesep + '\t') + os.linesep
        desc += '\tDefault batch size: %d' % self.batch_size + os.linesep
        desc += '\tDefault shuffle: %r' % self.shuffle
        desc += os.linesep + ')'

        return desc
