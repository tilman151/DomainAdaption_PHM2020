"""Custom layer definitions."""

import torch
import torch.nn as nn


class _GradientReverse(torch.autograd.Function):
    """Gradient reversal forward and backward definitions."""

    @staticmethod
    def forward(ctx, inputs):
        """Forward pass of gradient reversal."""
        return inputs

    @staticmethod
    def backward(ctx, grad):
        """Backward pass of gradient reversal."""
        return -grad


def gradient_reversal(x):
    """Perform gradient reversal on input."""
    return _GradientReverse.apply(x)


class GradientReversalLayer(nn.Module):
    """Module for gradient reversal."""

    def forward(self, inputs):
        return gradient_reversal(inputs)
