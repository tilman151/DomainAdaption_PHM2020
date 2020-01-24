"""Base class for trainers."""

import torch.nn as nn

import utils


class BaseTrainer(nn.Module):
    """Base class for trainers."""
    def __init__(self, optim, scheduler=None):
        """
        Construct a new trainer with specified optimizer.

        This is an abstract class that all trainers are children of. It specifies a
        setup function to connect the optimizer to a model before training. Child classes
        should implement the _step function that realizes a single optimization step given
        a batch of data.

        If you want to optimize only specific parameters of the model, you can override the
        _model_parameters function.

        An optional learning rate scheduler can be specified that will update each step.

        :param optim: class name string or class dict of the optimizer
        :param scheduler: class name string or class dict for learning rate optimizer
        """
        super(BaseTrainer, self).__init__()

        self.model = None

        self.optim = optim
        self.scheduler = scheduler

    def setup(self, model):
        """Setup the optimizer and scheduler with all model parameters."""
        self.model = model
        model_params = self._model_parameters()
        self.optim['parameters']['params'] = model_params
        self.optim = utils.build_object(self.optim)

        if self.scheduler is not None:
            self.scheduler['parameters']['optimizer'] = self.optim
            self.scheduler = utils.build_object(self.scheduler)

    def _model_parameters(self):
        """Return parameters to optimize."""
        return self.model.parameters()

    def _step(self, *batch):
        """Optimization step stub."""
        raise NotImplementedError()

    def forward(self, *batch):
        """Make an optimization and scheduler step."""
        results = self._step(*batch)
        if self.scheduler is not None:
            self.scheduler.step()

        return results
