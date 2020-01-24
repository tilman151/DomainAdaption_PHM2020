"""Model definitions for RUL estimation with domain adaption."""

import torch
import torch.nn as nn


class Conv(nn.Sequential):
    """Simple convolution layer with tanh activation."""

    def __init__(self, in_channels, out_channels, kernel_size):
        if kernel_size % 2 == 0:
            padding = (kernel_size // 2 - 1, kernel_size // 2)
        else:
            padding = (kernel_size // 2, kernel_size // 2)

        super(Conv, self).__init__(nn.ConstantPad1d(padding, 0.),
                                   nn.Conv1d(in_channels, out_channels, kernel_size),
                                   nn.Tanh())


class BaselineNetwork(nn.Module):
    """RUL estimation conv net."""

    def __init__(self, in_channels, seq_len, base_filters, num_layers, num_common_layers,
                 kernel_size, num_classes, dim_fc, dropout_rate):
        """
        Create a conv net for RUL estimation following https://doi.org/10.1016/j.ress.2017.11.021.

        :param in_channels: number of input channels
        :param seq_len: number of time steps in the sequence
        :param base_filters: number of filters in hidden conv layers
        :param num_layers: number of hidden layers
        :param kernel_size: kernel size of conv layers
        :param num_classes: number of output classes (1 for regression)
        :param dim_fc: number of units in fully connected layer
        :param dropout_rate: drop out rate after conv layers
        """
        super(BaselineNetwork, self).__init__()

        layers = [Conv(in_channels, base_filters, kernel_size)]
        for i in range(2, num_layers):
            layers.append(Conv(base_filters, base_filters, kernel_size))
        layers.append(Conv(base_filters, 1, 3))
        layers.append(nn.Sequential(nn.Flatten(),
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(seq_len, dim_fc),
                                    nn.Tanh()))
        layers.append(nn.Linear(dim_fc, num_classes))

        self.features = nn.Sequential(*layers)

        self.num_classes = num_classes
        self.num_common_layers = num_common_layers

        def _weight_init(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

        self.features[:-1].apply(_weight_init)
        nn.init.xavier_uniform_(self.features[-1].weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.features[-1].bias)

    def common_forward(self, inputs):
        """Run forward of feature extractor."""
        features = self.features[:self.num_common_layers](inputs)

        return features

    def adaption_forward(self, common_inputs):
        """Run adaption forward for MMD."""
        features = [common_inputs]
        for m in self.features[self.num_common_layers:]:
            features.append(m(features[-1]))

        return features[1:]

    def forward(self, inputs):
        """Classify data in target domain."""
        features = self.features(inputs)

        return features

    def source_forward(self, inputs):
        """Classify in source domain."""
        return self.forward(inputs)
