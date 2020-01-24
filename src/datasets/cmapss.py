"""Dataset for CMAPSS."""

import os
import torch.utils.data
import numpy as np
import sklearn.preprocessing as scalers

from datasets import BaseDataset, DATA_ROOT


def cmapss(fd, window_size, batch_size=1, shuffle=True, percent_fail_runs=None, percent_broken=None,
           normalization='minmax'):
    """CMAPSS construction function to get a BaseDataset."""
    train_data = CMAPSSDataset(fd, 'train', window_size=window_size, normalization=normalization,
                               percent_fail_runs=percent_fail_runs, percent_broken=percent_broken)
    test_data = CMAPSSDataset(fd, 'test', window_size=window_size, normalization=normalization,
                              percent_fail_runs=percent_fail_runs, percent_broken=percent_broken)

    return BaseDataset(train_data, test_data, batch_size, shuffle, dvc_file=os.path.join(DATA_ROOT, 'CMAPSS.dvc'))


def cmapss_hyperopt(fd, window_size, batch_size=1, shuffle=True, percent_fail_runs=None, percent_broken=None,
                    normalization='minmax'):
    """Build a dev and eval set from the training data to do hyperopt on."""
    train_data = CMAPSSDataset(fd, 'train', window_size=window_size, normalization=normalization,
                               percent_fail_runs=percent_fail_runs, percent_broken=percent_broken)
    split_idx = np.argwhere(train_data.targets == 1)
    num_train = int(0.8 * len(split_idx))

    dev_data = torch.utils.data.Subset(train_data, np.arange(split_idx[num_train]))
    eval_data = torch.utils.data.Subset(train_data, np.arange(split_idx[num_train], split_idx[-1] + 1))

    return BaseDataset(dev_data, eval_data, batch_size, shuffle, dvc_file=os.path.join(DATA_ROOT, 'CMAPSS.dvc'))


class CMAPSSDataset(torch.utils.data.Dataset):
    """Dataset class for the CMAPSS RUL prediction data."""

    def __init__(self, fd, split='train', max_rul=125, window_size=30,
                 percent_fail_runs=None, percent_broken=None,
                 channel_first=True, normalization=None, feature_select=True):
        super(CMAPSSDataset, self).__init__()

        assert split == 'train' or split == 'test', \
            'Split has to be train or test, but is %s' % split

        self.window_size = window_size
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.normalization = normalization
        self.feature_select = feature_select

        # Build path to feature file
        file_name = '%s_FD%03d.txt' % (split, fd)
        file_path = os.path.join(DATA_ROOT, 'CMAPSS', file_name)

        # Load and preprocess features
        self.features = self.load_features(file_path, feature_select,
                                           percent_fail_runs, percent_broken)

        # Normalize data
        if normalization is not None:
            if 'full_' in normalization:
                full = True
                normalization = normalization.replace('full_', '')
            else:
                full = False

            if normalization == 'minmax':
                self.normalize(file_path, scalers.MinMaxScaler(feature_range=(-1, 1)), full)
            elif normalization == 'z':
                self.normalize(file_path, scalers.StandardScaler(), full)
            elif normalization == 'robust':
                self.normalize(file_path, scalers.RobustScaler(), full)
            else:
                raise ValueError('Unknown normalization type: %s' % normalization)

        # Extract the time steps from the features
        time_steps = self._extract_time_steps()

        if split == 'train':
            # Build targets from time steps on training
            self.targets = self._generate_targets(time_steps, max_rul)
            # Window data to get uniform sequence lengths
            self._window_data(window_size)
        else:
            # Load targets from file on test
            self.targets = self._load_targets(fd, max_rul)
            # Crop data to get uniform sequence lengths
            self._crop_data(window_size)

        # Swap the channel dimension if needed
        if channel_first:
            self.features = self.features.transpose(0, 2, 1)

    @classmethod
    def load_features(cls, file_path, feature_select,
                      percent_fail_runs=None, percent_broken=None):
        """
        Load and preprocess the CMAPSS features.

        The function loads the features as a np.array from a file.
        The features can then be min-max normalized and cleaned of
        unneeded features. Afterwards the size of the dataset can be adjusted.
        Either the number of runs to failure can be reduced or the samples near
        failure for each run can be cut.

        :param file_path: path to feature file
        :param normalize: flag for normalization
        :param feature_select: flag for feature selection
        :param percent_fail_runs: percent of runs to keep
        :param percent_broken: percent of samples per run to cut
        :return: preprocessed features (np.array or list(np.array))
        """
        features = np.loadtxt(file_path)

        # Select features according to https://doi.org/10.1016/j.ress.2017.11.021
        if feature_select:
            feature_idx = [0, 1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
            features = features[:, feature_idx]

        # Split into runs
        _, samples_per_run = np.unique(features[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        features = np.split(features, split_idx, axis=0)

        # If loaded features are of the training split, truncate them
        if 'train' in os.path.basename(file_path):
            # Truncate the number of runs to failure
            if percent_fail_runs is not None and percent_fail_runs < 1:
                num_runs = int(percent_fail_runs * len(features))
                features = features[:num_runs]

            # Truncate the number of samples per run, starting at failure
            if percent_broken is not None and percent_broken > 0:
                for i, run in enumerate(features):
                    num_cycles = int(percent_broken * len(run))
                    run[:, 1] += num_cycles
                    features[i] = run[:-num_cycles]

        return features

    def _extract_time_steps(self):
        """Extract and return time steps from feature array."""
        time_steps = []
        for i, seq in enumerate(self.features):
            time_steps.append(seq[:, 1])
            seq = seq[:, 2:]
            self.features[i] = seq

        return time_steps

    def normalize(self, file_path, scaler, full):
        """Normalize features with sklearn transform."""
        if 'train' in os.path.basename(file_path):
            # Fit scaler on own data
            if full:
                # Use all data if full normalization
                train_features = self.load_features(file_path, self.feature_select)
                full_features = np.concatenate(train_features, axis=0)
            else:
                # Use only truncated data else
                full_features = np.concatenate(self.features, axis=0)
            scaler = scaler.fit(full_features[:, 2:])
        else:
            # Fit scaler on corresponding training split
            train_file = file_path.replace('test', 'train')
            percent_fail_runs = None if full else self.percent_fail_runs
            percent_broken = None if full else self.percent_broken
            train_features = self.load_features(train_file, self.feature_select,
                                                percent_fail_runs, percent_broken)
            full_features = np.concatenate(train_features, axis=0)
            scaler = scaler.fit(full_features[:, 2:])

        # Normalize features
        for i, run in enumerate(self.features):
            self.features[i][:, 2:] = scaler.transform(run[:, 2:])

    @staticmethod
    def _generate_targets(time_steps, max_rul):
        """Generate RUL targets from time steps."""
        return [np.minimum(max_rul, steps)[::-1] for steps in time_steps]

    @staticmethod
    def _load_targets(fd, max_rul):
        """Load target file."""
        file_name = 'RUL_FD%03d.txt' % fd
        file_path = os.path.join(DATA_ROOT, 'CMAPSS', file_name)
        targets = np.loadtxt(file_path)

        targets = np.minimum(max_rul, targets)

        return targets

    def _window_data(self, window_size):
        """Window features with specified window size."""
        new_features = []
        new_targets = []
        for seq, target in zip(self.features, self.targets):
            num_frames = seq.shape[0]
            seq = np.concatenate([np.zeros((window_size - 1, seq.shape[1])), seq])
            feature_windows = [seq[i:(i+window_size)] for i in range(0, num_frames)]
            new_features.extend(feature_windows)
            new_targets.append(target)

        self.features = np.stack(new_features, axis=0)
        self.targets = np.concatenate(new_targets)

    def _crop_data(self, window_size):
        """Crop length of features to specified window size."""
        for i, seq in enumerate(self.features):
            if seq.shape[0] < window_size:
                pad = (window_size - seq.shape[0], seq.shape[1])
                self.features[i] = np.concatenate([np.zeros(pad), seq])
            else:
                self.features[i] = seq[-window_size:]

        self.features = np.stack(self.features, axis=0)

    def __getitem__(self, index):
        """Return feature sample with target value."""
        return torch.tensor(self.features[index], dtype=torch.float32), \
               torch.tensor(self.targets[index], dtype=torch.float32)

    def __len__(self):
        """Return length of dataset."""
        return len(self.features)
