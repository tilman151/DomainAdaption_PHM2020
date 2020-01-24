"""Loss classes for Trainers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaximumMeanDiscrepancyLoss(nn.Module):
    """Class implementing the Maximum Mean Discrepancy Loss."""

    def __init__(self, num_kernels):
        """
        Create a new MMD loss module with n kernels.

        The maximum mean discrepancy loss is a similarity measure
        between two arbitrary distributions. The similarity is defined
        as the dot product in a reproducing Hilbert kernel space (RHKS)
        and is zero if and only if the distributions are identical.
        The RHKS is the space of the linear combination of multiple
        Gaussian kernels with bandwidths derived by the median heuristic.

        :param num_kernels: number of Gaussian kernels to use.
        """
        super(MaximumMeanDiscrepancyLoss, self).__init__()

        self.num_kernels = num_kernels
        self.betas = [1 / self.num_kernels] * self.num_kernels

    def forward(self, source_features, target_features):
        """
        Compute the MMD loss between two feature distributions.

        The MMD loss is computed as the sum of the MMD between
        the source and target feature distributions. For each of
        the feature pairs a bandwidth is computed via the median
        heuristic. The n Gaussian kernels will then use a bandwidth
        between median / (2 ** (n / 2)) and median * (2 ** (n / 2)).

        :param source_features: list of source feature tensors of shape [batch, feats]
        :param target_features: list of target feature tensors of shape [batch, feats]
        :return: scalar sum of MMD loss for each pair of list elements
        """
        batch_size = source_features[0].shape[0]
        disc = 0
        for s_feat, t_feat in zip(source_features, target_features):
            feats = torch.cat((s_feat, t_feat), dim=0)
            distances = self._pairwise_distances(feats, feats)

            n_samples = 2 * batch_size
            gammas = self._get_gamma(distances, n_samples)
            distances = self._multi_kernel(distances, gammas)

            s2s = distances[:batch_size, :batch_size]
            t2t = distances[batch_size:, batch_size:]
            s2t = distances[:batch_size, batch_size:]

            disc += torch.mean(s2s + t2t - 2 * s2t)

        return disc

    def _get_gamma(self, distances, n_samples):
        """Compute gammas for n Gaussian kernels via median heuristic."""
        bandwidth = float(n_samples ** 2 - n_samples) / torch.sum(distances.detach())
        bandwidth /= 2 ** (self.num_kernels // 2)
        gammas = [bandwidth * (2 ** i) for i in range(self.num_kernels)]
        return gammas

    @staticmethod
    def _pairwise_distances(x, y):
        """Compute pairwise linear distances between features."""
        num_elems = x.shape[0]
        x = x.view(num_elems, 1, -1)
        y = y.view(1, num_elems, -1)
        distances = (x - y) ** 2

        return distances.sum(-1)

    @staticmethod
    def _multi_kernel(distances, gammas):
        """Compute Gaussian kernel for linear distances."""
        kernels = [torch.exp(-distances * gamma) for gamma in gammas]

        return sum(kernels) / len(gammas)


class JointMaximumMeanDiscrepancyLoss(MaximumMeanDiscrepancyLoss):
    """Class for the Joint Maximum Mean Discrepancy Loss."""

    def __init__(self):
        """
        Create a new JMMD loss module.

        The JMMD loss captures the interactions between multiple
        pairs of distributions by multiplying their distances in
        the RHKS before computing MMD. A single Gaussian kernel is
        used for the RHKS.
        """
        super(JointMaximumMeanDiscrepancyLoss, self).__init__(num_kernels=1)

    def forward(self, source_features, target_features):
        """
        Compute the JMMD loss between a list of pairs of feature distributions.

        The JMMD loss is computed as the MMD of
        the product of the distances of source and target feature distributions.
        For each of the feature pairs a bandwidth is computed via the median
        heuristic.

        :param source_features: list of source feature tensors of shape [batch, feats]
        :param target_features: list of target feature tensors of shape [batch, feats]
        :return: scalar JMMD loss for the feature distributions
        """
        batch_size = source_features[0].shape[0]
        distances = []

        for source, target in zip(source_features, target_features):
            feats = torch.cat([source, target], dim=0)
            dist = self._pairwise_distances(feats, feats)
            n_samples = 2 * batch_size
            gammas = self._get_gamma(dist, n_samples)
            distances.append(self._multi_kernel(dist, gammas))

        distances = torch.stack(distances, dim=0).prod(0)  # [2*batch, 2*batch]
        s2s = distances[:batch_size, :batch_size]  # [batch, batch]
        t2t = distances[batch_size:, batch_size:]  # [batch, batch]
        s2t = distances[:batch_size, batch_size:]  # [batch, batch]

        disc = torch.mean(s2s + t2t - 2 * s2t)

        return disc


class EntropyLoss(nn.Module):
    """Class for the Shannon entropy."""

    def forward(self, inputs):
        """Compute the Shannon entropy of the input vector."""
        return torch.mean(-1. * torch.sum(F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1), dim=-1))


class KLDivGaussianLoss(nn.Module):
    """Kulback-Leibler divergence loss."""

    def forward(self, mu, log_var):
        """Compute KL loss for multivariate Gaussian."""
        loss = 1 + log_var - log_var.exp() - mu.pow(2)
        loss = -0.5 * loss.sum()

        return loss


class RMSELoss(nn.Module):
    """Root Mean Squared error loss."""

    def forward(self, inputs, targets):
        """Compute the root of the mean squared error."""
        return torch.sqrt(F.mse_loss(inputs, targets, reduction='mean'))
