"""Base class for machine learning tasks."""

import warnings
import copy

import summary
import utils

from metrics import BaseMetric


class BaseTask:
    """Class for a machine learning task."""

    def __init__(self, name, device, dataset_cfg, model_cfg, trainer_cfg, metrics=None):
        """
        Set up a training environment defined by a dataset, model, trainer and metrics.

        This initializes all needed components for a learning task.
        The configurations for the components are dictionaries intended
        for the utils.build_object function (see there for more info).
        Additionally a summary writer is constructed that will log summaries
        to TensorBoard and scalars to MLFlow. After finishing training, the
        TensorBoard event file is added as an artifact to MLFlow.

        :param name: name of the task
        :param device: device to train on (e.g. cpu or cuda)
        :param dataset_cfg: config dict for dataset
        :param model_cfg: config dict for model
        :param trainer_cfg: config dict for trainer
        :param metrics: list of metrics class name strings to use for evaluation
        """
        print('Set up summaries...')
        self.summary_writer = summary.SummaryWriter(name)
        print(self.summary_writer)
        print('#' * 40)

        self.device = device
        print('Build dataset...')
        self.dataset = utils.build_object(copy.deepcopy(dataset_cfg))
        print(self.dataset)
        print('#' * 40)

        print('Build model...')
        self.model = utils.build_object(copy.deepcopy(model_cfg))
        self.model = self.model.to(self.device)
        print(self.model)
        print('#' * 40)

        print('Build trainer...')
        self.trainer = utils.build_object(copy.deepcopy(trainer_cfg))
        self.trainer = self.trainer.to(self.device)
        print('#' * 40)

        self.epoch = None

        print('Build metrics...')
        self.metrics = self._build_metrics(metrics)
        [print(metric) for metric in self.metrics]
        print('#' * 40)

        self.trainer.setup(self.model)

    def _build_metrics(self, metrics):
        """Build the metric objects from strings."""
        if metrics is not None:
            if not isinstance(metrics, list):
                metrics = [metrics]
            for metric in metrics:
                metric['parameters'] = {'device': self.device}
            metrics = [utils.build_object(metric) for metric in metrics]
            assert all(isinstance(metric, BaseMetric) for metric in metrics), \
                'One element in the metric list is not a BaseMetric'
        else:
            metrics = []
            warnings.warn('No metrics defined for this task.', UserWarning)

        return metrics

    def train(self, epochs, num_data_workers=None):
        """
        Train for the specified number of epochs.

        The task will iterate over the training data and
        move all elements of the batch to the device. The
        batch is then passed to the forward function of the
        trainer to execute the training step. The results
        of the step are written to summary every tenth step
        and to console ten times per epoch. After iterating over
        the training data, the model will be passed to each metric
        with the evaluation data. The results are again written to summary.
        Afterwards the model is saved as a snapshot to file.
        This process is repeated for each epoch.

        :param epochs: number of epochs to train
        :param num_data_workers: number of worker threads for loading data
        """
        step = 0
        training_data = self.dataset.training(num_workers=num_data_workers)
        batch_per_epoch = len(training_data)
        print_step = max(1, batch_per_epoch // 10)

        print('Training on %s...' % self.device)
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            self.model.train()
            for i, batch in enumerate(training_data):
                batch = [tensor.to(self.device, non_blocking=True) for tensor in batch]
                scalar, result_dict = self.trainer(*batch)

                if step % 10 == 0:
                    self.summary_writer.write_results(result_dict, step, scalar_tab='training')
                if step % print_step == 0:
                    print('Step %d - Epoch %d - Batch %d/%d:\t%.5f' %
                          (step, epoch, i, batch_per_epoch, scalar))

                step += 1

            self.summary_writer.save(self.model, 'model', str(epoch))
            status_results = self.eval(self.model, num_data_workers)

        self.summary_writer.close()

    def eval(self, model, num_data_workers):
        """
        Evaluate the model on the evaluation data with each metric.

        :param model: model to evaluate.
        :param num_data_workers: number of worker threads for loading data
        :return: status results dict of the metrics
        """
        status_results = {}
        print('Evaluation...')
        for metric in self.metrics:
            eval_data = self.dataset.evaluation(shuffle=False, num_workers=num_data_workers)
            status, result_dict = metric.run(model, eval_data)

            self.summary_writer.write_results(result_dict, self.epoch, scalar_tab='evaluation')
            if isinstance(status, float):
                print('\tEpoch %d - Eval metric %s: %.5f' % (self.epoch, metric.name, status))
            else:
                print('\tEpoch %d - Eval metric %s: %s' % (self.epoch, metric.name, str(status)))

            status_results[metric.name] = status

        return status_results
