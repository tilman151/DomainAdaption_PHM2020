"""Trainers for regression tasks."""

from trainers import losses
from . import domain_adaption

from trainers import BaseTrainer


class RMSETrainer(BaseTrainer):
    """Class for training supervised regression models with RMSE."""

    def __init__(self, *args, **kwargs):
        super(RMSETrainer, self).__init__(*args, **kwargs)

        self.criterion = losses.RMSELoss()

    def _step(self, inputs, targets):
        """Make one supervised training step."""
        self.optim.zero_grad()

        outputs = self.model(inputs)

        loss = self.criterion(outputs.squeeze(), targets)
        loss.backward()

        self.optim.step()

        results = {'scalars': {'rmse': loss.item()}}
        return loss.item(), results


class DANTrainer(domain_adaption.AbstractDANTrainer):
    """Class for a Domain Adaption Network regression trainer."""

    def __init__(self, optim, lmbd, **kwargs):
        """
        Create a new DAN trainer for regression.

        :param optim: class name string of the optimizer
        :param lmbd: influence of MMD loss on training
        :param kwargs: kwargs for BaseTrainer
        """
        super(DANTrainer, self).__init__(optim, lmbd, **kwargs)

        self._criterionRMSE = losses.RMSELoss()

    @property
    def criterionMain(self):
        return self._criterionRMSE

    @property
    def main_loss_name(self):
        return 'rmse'


class RTNTrainer(domain_adaption.AbstractRTNTrainer):
    """Class for a Residual Transfer Network regression trainer."""

    def __init__(self, optim, mmd_lmbd, entropy_lmbd, scale_lr=False, caffe_version=False, **kwargs):
        """
        Create a new RTN trainer for regression.

        :param optim: class name string of the optimizer
        :param mmd_lmbd: influence of MMD loss on training
        :param entropy_lmbd: influence of target prediction entropy
        :param scale_lr: flag for multiplying lr for classification layers by ten
        :param caffe_version: use caffe version of tensor product
        :param kwargs: kwargs for BaseTrainer
        """
        super(RTNTrainer, self).__init__(optim, mmd_lmbd, entropy_lmbd, scale_lr, caffe_version, **kwargs)

        self._criterionRMSE = losses.RMSELoss()

    @property
    def criterionMain(self):
        return self._criterionRMSE

    @property
    def main_loss_name(self):
        return 'rmse'


class JANTrainer(domain_adaption.AbstractJANTrainer):
    """Class for Joint Adaption Network regression trainer."""

    def __init__(self, optim, max_lmbd, max_steps, **kwargs):
        """
        Create a new JAN trainer for regression.

        :param optim: class name string of the optimizer
        :param max_steps: maximum optimization steps (to compute lambda)
        :param kwargs: kwargs for BaseTrainer
        """
        super(JANTrainer, self).__init__(optim, max_lmbd, max_steps, **kwargs)

        self._criterionRMSE = losses.RMSELoss()

    @property
    def criterionMain(self):
        return self._criterionRMSE

    @property
    def main_loss_name(self):
        return 'rmse'


class DomainAdversarialTrainer(domain_adaption.AbstractDomainAdversarialTrainer):
    """Class for Domain Adversarial Adaption Network regression trainer."""

    def __init__(self, optim, lmbd, disc_num_layers, disc_in_units, disc_num_units, act, **kwargs):
        """
        Create a new DAAN trainer for regression.

        :param optim: class name string of the optimizer
        :param lmbd: influence of domain discriminator
        :param disc_num_layers: number of layers for the domain discriminator
        :param disc_in_units: number of input units from network
        :param disc_num_units: number of units per layer
        :param disc_act: activation function for domain discriminator
        :param kwargs: kwargs for BaseTrainer
        """
        super(DomainAdversarialTrainer, self).__init__(optim, lmbd, disc_num_layers, disc_in_units,
                                                       disc_num_units, act, **kwargs)

        self._criterionRMSE = losses.RMSELoss()

    @property
    def criterionMain(self):
        return self._criterionRMSE

    @property
    def main_loss_name(self):
        return 'rmse'
