"""Abstract trainers for domain adaption."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers import losses
from models import layers

from trainers import BaseTrainer
from summary import SummaryWriter


class AbstractDANTrainer(BaseTrainer):
    """Abstract class for a Domain Adaption Network regression trainer."""

    def __init__(self, optim, lmbd, **kwargs):
        """
        Create a new abstract DAN trainer.

        To use this trainer for a specific task, one has to sub-class it and set the
        main loss (e.g. RMSE for regression). This is done by overriding the
        property abstract functions.

        :param optim: class name string of the optimizer
        :param lmbd: influence of MMD loss on training
        :param kwargs: kwargs for BaseTrainer
        """
        super(AbstractDANTrainer, self).__init__(optim, **kwargs)

        self.lmbd = lmbd
        self.criterionMMD = losses.MaximumMeanDiscrepancyLoss(5)

    @property
    def criterionMain(self):
        """Return the main loss function for this trainer."""
        return NotImplementedError

    @property
    def main_loss_name(self):
        """Return the name to be logged for the main loss."""
        return NotImplementedError

    def _step(self, source_inputs, source_targets, target_inputs, target_targets):
        """
        Make on optimization step.

        First the pre-trained feature layers are evaluated for source and target
        data. Then the output of each fully-connected classification layer is used
        to compute the MMD loss. The final loss is the main loss of the source
        predictions plus the MMD loss.

        :param source_inputs: source data
        :param source_targets: source data target values
        :param target_inputs: target data
        :param target_targets: target data values (not used)
        :return: scalar total loss and dictionary of scalar loss summaries
        """
        self.optim.zero_grad()

        common_inputs = torch.cat([source_inputs, target_inputs], dim=0)
        common_features = self.model.common_forward(common_inputs)
        common_features = self.model.adaption_forward(common_features)
        common_features = [torch.split(features, source_inputs.shape[0], dim=0) for features in common_features]
        source_features, target_features = zip(*common_features)

        loss_main = self.criterionMain(source_features[-1].squeeze(), source_targets)
        loss_mmd = self.criterionMMD(source_features, target_features)
        loss = loss_main + self.lmbd * loss_mmd
        loss.backward()

        self.optim.step()

        results = {'scalars': {'loss': loss.item(),
                               self.main_loss_name: loss_main.item(),
                               'max_mean_disc': loss_mmd.item()}}
        return loss.item(), results


class AbstractRTNTrainer(BaseTrainer):
    """Abstract class for a Residual Transfer Network trainer."""

    def __init__(self, optim, mmd_lmbd, entropy_lmbd, scale_lr=False, caffe_version=False, **kwargs):
        """
        Create a new abstract RTN trainer.

        To use this trainer for a specific task, one has to sub-class it and set the
        main loss (e.g. RMSE for regression). This is done by overriding the
        property abstract functions.

        :param optim: class name string of the optimizer
        :param mmd_lmbd: influence of MMD loss on training
        :param entropy_lmbd: influence of target prediction entropy
        :param scale_lr: flag for multiplying lr for classification layers by ten
        :param caffe_version: use caffe version of tensor product
        :param kwargs: kwargs for BaseTrainer
        """
        super(AbstractRTNTrainer, self).__init__(optim, **kwargs)

        self.mmd_lmbd = mmd_lmbd
        self.entropy_lmbd = entropy_lmbd
        self.scale_lr = scale_lr
        self.caffe_version = caffe_version
        self.criterionMMD = losses.MaximumMeanDiscrepancyLoss(1)
        self.criterionH = losses.EntropyLoss()

    @property
    def criterionMain(self):
        """Return the main loss function for this trainer."""
        return NotImplementedError

    @property
    def main_loss_name(self):
        """Return the name to be logged for the main loss."""
        return NotImplementedError

    @staticmethod
    def _tensor_product(x, y):
        """Compute outer product."""
        outer = torch.einsum('bi,bj->bij', x, y)
        return outer

    def _model_parameters(self):
        """Multiply lr by ten for classification layers if flag is set."""
        if self.scale_lr:
            finetuning_params = list(self.model.alex.parameters()) + \
                                list(self.model.alex_classifier.parameters())
            training_params = list(self.model.bottleneck.parameters()) + \
                              list(self.model.classifier.parameters()) + \
                              list(self.model.delta_classifier.parameters())

            lr = self.optim_params['lr']
            parameters = [{'params': finetuning_params, 'lr': lr},
                          {'params': training_params, 'lr': 10*lr}]
        else:
            parameters = super(AbstractRTNTrainer, self)._model_parameters()

        return parameters

    def _step(self, source_inputs, source_targets, target_inputs, target_targets):
        """
        Make on optimization step.

        First the pre-trained feature layers are evaluated for source and target
        data. Then the output of each fully-connected classification layer is used
        to compute the tensor product. This is then used to compute the MMD loss.
        The final loss is the cross entropy of the source predictions plus the MMD
        loss plus the entropy of the target predictions.

        :param source_inputs: source data
        :param source_targets: source data target values
        :param target_inputs: target data
        :param target_targets: target data values (not used)
        :return: scalar total loss and dictionary of scalar loss summaries
        """
        self.optim.zero_grad()

        common_inputs = torch.cat([source_inputs, target_inputs], dim=0)
        common_features = self.model.common_forward(common_inputs)
        common_features = self.model.adaption_forward(common_features)
        common_features = [torch.split(features, source_inputs.shape[0], dim=0) for features in common_features]
        source_features, target_features = zip(*common_features)

        source_pred = self.model.delta_forward(source_features[-1])
        loss_main = self.criterionMain(source_pred.squeeze(), source_targets)

        source_tensor = self._tensor_product(*source_features).flatten(start_dim=1)
        target_tensor = self._tensor_product(*target_features).flatten(start_dim=1)
        if self.caffe_version:
            common_tensor = torch.cat([source_tensor, target_tensor])
            common_tensor = common_tensor.reshape(common_tensor.shape[0], 1, self.model.bottleneck_dim, -1)
            common_tensor = F.max_pool2d(common_tensor, kernel_size=2, stride=2)
            common_tensor = common_tensor.reshape(common_tensor.shape[0] // 2, -1)
            source_tensor, target_tensor = torch.split(common_tensor, common_tensor.shape[0] // 2, dim=0)
        loss_mmd = self.criterionMMD([source_tensor], [target_tensor])

        loss = loss_main + self.mmd_lmbd * loss_mmd
        loss.backward()

        self.optim.step()

        results = {'scalars': {'loss': loss.item(),
                               self.main_loss_name: loss_main.item(),
                               'max_mean_disc': loss_mmd.item()}}
        return loss.item(), results


class AbstractJANTrainer(BaseTrainer):
    """Abstract class for Joint Adaption Network trainer."""

    def __init__(self, optim, max_lmbd, max_steps, **kwargs):
        """
        Create a new abstract JAN trainer.

        To use this trainer for a specific task, one has to sub-class it and set the
        main loss (e.g. RMSE for regression). This is done by overriding the
        property abstract functions.

        :param optim: class name string of the optimizer
        :param max_steps: maximum optimization steps (to compute lambda)
        :param kwargs: kwargs for BaseTrainer
        """
        super(AbstractJANTrainer, self).__init__(optim, **kwargs)

        self.criterionJMMD = losses.JointMaximumMeanDiscrepancyLoss()

        self.step = 0
        self.max_lmbd = max_lmbd
        self.max_steps = max_steps
        self.summary = SummaryWriter('')

    @property
    def criterionMain(self):
        """Return the main loss function for this trainer."""
        return NotImplementedError

    @property
    def main_loss_name(self):
        """Return the name to be logged for the main loss."""
        return NotImplementedError

    @property
    def _lmbd(self):
        """Return current value of lambda and increment step counter."""
        p = self.step / self.max_steps
        lmbd = 2. / (1. + math.exp(-10. * p)) - 1.
        self.step += 1

        return lmbd * self.max_lmbd

    def _step(self, source_inputs, source_targets, target_inputs, target_targets):
        """
        Make on optimization step.

        First the pre-trained feature layers are evaluated for source and target
        data. Then the output of each fully-connected classification layer is used
        to compute the JMMD loss. The final loss is the cross entropy of the source
        predictions plus the JMMD loss. The influence of the latter is determined by
        the training process.

        :param source_inputs: source data
        :param source_targets: source data target values
        :param target_inputs: target data
        :param target_targets: target data values (not used)
        :return: scalar total loss and dictionary of scalar loss summaries
        """
        self.optim.zero_grad()

        common_inputs = torch.cat([source_inputs, target_inputs], dim=0)
        common_features = self.model.common_forward(common_inputs)
        common_features = self.model.adaption_forward(common_features)
        common_features = [torch.split(features, source_inputs.shape[0], dim=0) for features in common_features]
        source_features, target_features = zip(*common_features)

        loss_main = self.criterionMain(source_features[-1].squeeze(), source_targets)
        loss_mmd = self.criterionJMMD(source_features, target_features)
        lmbd = self._lmbd
        loss = loss_main + lmbd * loss_mmd
        loss.backward()

        self.optim.step()

        if self.step % 10 == 0:
            self.summary.add_scalar('training/jmmd_lambda', lmbd, global_step=self.step)

        results = {'scalars': {'loss': loss.item(),
                               self.main_loss_name: loss_main.item(),
                               'max_mean_disc': loss_mmd.item()}}
        return loss.item(), results


class AbstractDomainAdversarialTrainer(BaseTrainer):
    """Abstract class for Domain Adversarial Adaption Network trainer."""

    def __init__(self, optim, lmbd, disc_num_layers, disc_in_units, disc_num_units, disc_act, **kwargs):
        """
        Create a new abstract DAAN trainer.

        To use this trainer for a specific task, one has to sub-class it and set the
        main loss (e.g. RMSE for regression). This is done by overriding the
        property abstract functions.

        :param optim: class name string of the optimizer
        :param lmbd: influence of domain discriminator
        :param disc_num_layers: number of layers for the domain discriminator
        :param disc_in_units: number of input units from network
        :param disc_num_units: number of hidden units per layer
        :param disc_act: activation function for domain discriminator
        :param kwargs: kwargs for BaseTrainer
        """
        super(AbstractDomainAdversarialTrainer, self).__init__(optim, **kwargs)

        self.num_layers = disc_num_layers
        self.num_units = disc_num_units
        self.lmbd = lmbd
        self.act = eval(disc_act['name'])
        self.criterionCE = nn.BCEWithLogitsLoss(reduction='mean')

        domain_disc = [layers.GradientReversalLayer(), nn.Flatten(start_dim=1)]
        domain_disc += [nn.Linear(disc_in_units, disc_num_units), self.act()]
        domain_disc += [nn.Linear(disc_num_units, disc_num_units), self.act()] * (disc_num_layers - 1)
        domain_disc += [nn.Linear(disc_num_units, 1)]
        self.domain_disc = nn.Sequential(*domain_disc)

    @property
    def criterionMain(self):
        """Return the main loss function for this trainer."""
        return NotImplementedError

    @property
    def main_loss_name(self):
        """Return the name to be logged for the main loss."""
        return NotImplementedError

    def _model_parameters(self):
        """Return parameters of model and domain discriminator."""
        model_params = super(AbstractDomainAdversarialTrainer, self)._model_parameters()
        model_params = list(model_params) + list(self.domain_disc.parameters())

        return model_params

    def _step(self, source_inputs, source_targets, target_inputs, target_targets):
        self.optim.zero_grad()

        batch_size = source_inputs.shape[0]
        common_inputs = torch.cat([source_inputs, target_inputs], dim=0)
        common_features = self.model.common_forward(common_inputs)
        source_features = common_features[:batch_size]

        source_features = self.model.adaption_forward(source_features)
        source_prediction = source_features[-1].squeeze()
        loss_main = self.criterionMain(source_prediction, source_targets)

        common_disc = self.domain_disc(common_features)
        domain_labels = torch.cat([torch.zeros(batch_size, 1, device=common_disc.device),
                                   torch.ones(batch_size, 1, device=common_disc.device)])
        loss_adv = self.criterionCE(common_disc, domain_labels)

        loss = loss_main + self.lmbd * loss_adv
        loss.backward()

        self.optim.step()

        results = {'scalars': {'loss':              loss.item(),
                               self.main_loss_name: loss_main.item(),
                               'domain_adv':        loss_adv.item()}}
        return loss.item(), results
