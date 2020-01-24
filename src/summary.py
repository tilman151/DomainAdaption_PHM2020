"""Summary writing convenience utils."""

import os
import torch
import torch.utils.tensorboard
import numpy as np
import matplotlib.pyplot as plt
import warnings
import mlflow

import utils

LOG_ROOT = "./results"


class Singleton(type):
    """Meta class for singletons."""

    _instances = {}  # instance dictionary

    def __call__(cls, *args, **kwargs):
        """Construct new instance or return existing one."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SummaryWriter(torch.utils.tensorboard.SummaryWriter, metaclass=Singleton):
    """Singleton extension of TensorBoardX summary writer."""
    def __init__(self, logdir, **kwargs):
        """
        Get the SummaryWriter singleton instance or create a new one if not exisiting yet.

        This class writes summaries to TensorBoard and MLFlow. It is designed as a singleton,
        so that each part of the program uses the same instance. Once constructed the constructor
        always returns the same instance regardless of parameters passed.

        The summary writer will not write to disk when debugging is detected to avoid cluttering
        the results directory.

        :param logdir: name of log directory
        :param kwargs: kwargs for underlying TensorBoardX summary writer.
        """
        if not os.path.exists(LOG_ROOT):
            os.makedirs(LOG_ROOT)

        dirs = sorted(os.listdir(LOG_ROOT), reverse=True)
        num = 0
        for d in dirs:
            if logdir in d:
                num = int(d[:3]) + 1
                break

        logdir = str(num).zfill(3) + '_' + logdir
        logdir = os.path.join(LOG_ROOT, logdir)

        if utils.is_debugging():
            warnings.warn('Debugging mode: will write to temporary TensorBoard file.', UserWarning)
            logdir = utils.build_tmp_dir()

        super(SummaryWriter, self).__init__(logdir, flush_secs=60, **kwargs)

    def __str__(self):
        """Return string representation."""
        desc = 'SummaryWriter(' + os.linesep
        desc += '\tlogdir: %s' % self.log_dir
        desc += os.linesep + ')'

        return desc

    def write_results(self, results, time_step, scalar_tab=''):
        """
        Write dictionary of results to TensorBoard and MLFlow.

        This convenience function takes a dictionary that specifies
        several different summaries to be written. The summaries will
        be passed to the right add_* function of the summary writer.
        Scalars will be logged to MLFlow, too.

        example: {'scalars': {'metric1': 0.1, 'metric2': 0.5}, 'images': {'img1': img_tensor}}

        :param results: dictionary of summaries
        :param time_step: time step to log for
        :param scalar_tab: prefix to use right tab in scalar overview in TensorBoard
        """
        if not scalar_tab.endswith('/'):
            scalar_tab += '/'

        if 'scalars' in results:
            for tag, scalar in results['scalars'].items():
                self.add_scalar(tag=scalar_tab + tag,
                                scalar_value=scalar,
                                global_step=time_step)
                if not utils.is_debugging():
                    mlflow_tag = (scalar_tab + tag).replace('/', '_')
                    mlflow.log_metric(mlflow_tag, scalar, time_step)

        if 'images' in results:
            for tag, image in results['images'].items():
                if image.dim() == 2:
                    formats = 'HW'
                elif image.dim() == 3:
                    formats = 'CHW' if image.shape[0] in [1, 3] else 'HWC'
                elif image.dim() == 4:
                    formats = 'NCHW' if image.shape[1] in [1, 3] else 'NHWC'
                else:
                    raise ValueError('Unknown image format with shape %s' % str(image.shape))

                self.add_images(tag, image, time_step, dataformats=formats)

        if 'series' in results:
            for tag, series in results['series'].items():
                plot = self._plot_series(tag, series)
                self.add_figure(tag, plot, time_step, close=True)

        if 'embeddings' in results:
            for tag, embedding in results['embeddings'].items():
                self.add_embedding(**embedding, tag=tag, global_step=time_step)

    @staticmethod
    def _plot_series(tag, series):
        """Plot a time series with matplotlib."""
        if type(series) == torch.Tensor:
            series = series.detach().cpu().numpy()

        if len(series.shape) == 1:
            series = np.expand_dims(series, axis=1)

        figure = plt.figure()
        plt.title(tag)
        x = np.arange(series.shape[0])
        for y in series:
            plt.plot(x, y)

        return figure

    def save(self, obj, name, tag):
        """
        Save a model to file.

        The model is saved to the log directory as 'name_tag.pth'

        :param obj: model to save
        :param name: name of the save file
        :param tag: prefix tag
        """
        if utils.is_debugging():
            warnings.warn('Debugging mode: will save model checkpoint to temporary dir.', UserWarning)

        file_name = os.path.join(self.log_dir, name + '_' + tag + '.pth')
        torch.save(obj, file_name)

    def close(self):
        """Close the event file and add it with the model files to MLFlow."""
        super().close()

        if not utils.is_debugging():
            files = os.listdir(self.log_dir)
            event_file = [f for f in files if f.startswith('events')][0]
            model_files = [f for f in files if f.endswith('.pth')]

            mlflow.log_artifact(os.path.join(self.log_dir, event_file), artifact_path='events')
            for m in model_files:
                mlflow.log_artifact(os.path.join(self.log_dir, m), artifact_path='models')
