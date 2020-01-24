"""Main run file for dnnlib."""
import argparse
import mlflow
import torch
import torch.backends.cudnn
import numpy as np
import os
import warnings

import utils

from tasks import BaseTask


def run(config, device, epochs, replications, seed, num_data_workers):
    """
    Run an experiment of the given config.

    A MLFlow experiment will be set according to
    the name in the config. A BaseTask will be build
    and the train function called. Each call of the run function
    with the same config will be a run of this experiment.
    If replications is set to a number bigger than one, a nested
    run is created and the task executed this number of times.

    When debugging, nothing is written to disk to avoid
    cluttering the results directory.

    :param config: path to the config JSON file or config dict
    :param device: device to train on
    :param epochs: epochs to train for
    :param replications: number of times to replicate this run
    :param seed: random seed to use
    :param num_data_workers: number of worker threads for data loading
    """
    # Set seed for randomization
    if seed is not None:
        # Make PyTorch and numpy deterministic
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        print('Fixed randomization. Seed %d' % seed)
        print('#' * 40)
    else:
        # Retrieve default seed, as it is not set
        seed = np.random.randint(np.iinfo(np.int32).max)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load config JSON
    if isinstance(config, str):
        print('Run experiment from %s' % config)
        print('#' * 40)
        config = utils.read_config(config)
    elif isinstance(config, dict):
        print('Run experiment with dict named %s' % config['name'])
        print('#' * 40)
    else:
        raise ValueError('Config has to be either a string path or a dict, but is %s.' %
                         str(type(dict)))

    # Extract config dicts for components
    name = config['name']
    dataset = config['dataset']
    model = config['model']
    trainer = config['trainer']
    metrics = config['metrics']

    # Setup mlflow experiment
    if utils.is_debugging():
        # Reroute mlflow to tmp file on debugging
        warnings.warn('Debugging mode: MLFlow stuff will be saved to temporary dir.', UserWarning)
        mlflow.set_tracking_uri('file:' + utils.build_tmp_dir())
    else:
        script_path = os.path.dirname(__file__)
        root_path = os.path.dirname(script_path)
        mlflow.set_tracking_uri('file:' + root_path)
    mlflow.set_experiment(name)

    # Start the top level run
    nest_runs = True if replications > 0 else False
    with mlflow.start_run(nested=nest_runs):
        # Log parameters to run
        utils.log_config(config)
        mlflow.log_param('max_epochs', epochs)
        mlflow.log_param('seed', seed)
        mlflow.set_tag('device', device)

        if nest_runs:
            # Open child runs for each replication
            mlflow.log_param('replications', replications)
            seeds = np.random.randint(np.iinfo(np.int32).max, size=replications)
            for i, s in enumerate(seeds):
                print('Run replication %d/%d...' % (i, replications))
                with mlflow.start_run(nested=True):
                    # Log params to child runs
                    utils.log_config(config)
                    mlflow.set_tag('replication', i)

                    # Set derived seed for child runs to make each reproducible
                    mlflow.log_param('seed', s)
                    torch.manual_seed(s)
                    np.random.seed(s)

                    # Execute run
                    task = BaseTask(name, device, dataset, model, trainer, metrics)
                    task.train(epochs, num_data_workers)
        else:
            # Simply execute top level run, when replications are zero
            task = BaseTask(name, device, dataset, model, trainer, metrics)
            task.train(epochs, num_data_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment configured by a JSON file.')
    parser.add_argument('config', help='Path to config JSON file')
    parser.add_argument('-d', '--device', default='cpu', help='Device to train on')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('-r', '--replications', type=int, default=0, help='Number of times to replicate this run')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed to make things deterministic')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Number of workers to load data')
    opt = parser.parse_args()

    run(opt.config, opt.device, opt.epochs, opt.replications, opt.seed, opt.workers)
