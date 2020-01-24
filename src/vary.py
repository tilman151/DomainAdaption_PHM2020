"""Script for varying hyperparameters of a config."""
import argparse
import sklearn.model_selection

from easydict import EasyDict

import utils

from run import run


def modify_config(config, mods):
    """Modify hyperparameters in a config dict."""
    # Cast dict to EasyDict to get dot notation access
    config = EasyDict(config)
    # For each hparam in dot notation
    for name, value in mods.items():
        # Build assignment string and execute it
        exec('config.' + name + '=' + str(value))

    return config


def vary_hyperparameters(config, hyperparams, device, epochs, replications, seed, num_data_workers):
    """
    Run a config with all combinations of hyperparameters specified.

    The script takes the same parameters as run.py and an additional
    list of hyperparameters to vary over. It will span a grid of these
    hyperparameters and execute run.py with a modified config dict for
    each combination of hyperparameter values on the grid.

    :param config: path to config JSON file
    :param hyperparams: list of hyperparams as string ('hparam.in.dot.notation=['list','of','values']')
    :param device: device to train on
    :param epochs: number of epochs to train
    :param replications: number of replications for each run
    :param seed: random seed passed to run.py
    :param num_data_workers: number of data loading processes
    """
    config = utils.read_config(config)

    # Parse list of hyperparameters from CLI
    grid_axis = {}
    for param in hyperparams:
        param, values = param.split('=')
        values = eval(values)
        grid_axis[param] = values
    # Create hparam grid
    hyperparams = sklearn.model_selection.ParameterGrid(grid_axis)

    # Execute run.py for each modified config
    for params in hyperparams:
        config = modify_config(config, params)
        run(config, device, epochs, replications, seed, num_data_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment configured by a JSON file.')
    parser.add_argument('config', help='Path to config JSON file')
    parser.add_argument('hyperparams', nargs='+', metavar='PARAM=VALUES', help='List of hyperparameters to vary over')
    parser.add_argument('-d', '--device', default='cpu', help='Device to train on')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('-r', '--replications', type=int, default=0, help='Number of times to replicate this run')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed to make things deterministic')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Number of workers to load data')
    opt = parser.parse_args()

    vary_hyperparameters(opt.config, opt.hyperparams, opt.device, opt.epochs, opt.replications, opt.seed, opt.workers)
