"""Utility functions."""

import importlib
import inspect
import mlflow
import tempfile
import atexit
import shutil
import json


def build_object(config):
    """
    Return an instance of a class specified by name and parameters.

    The function takes a dictionary of a fully specified class name
    and key word parameters. It will then try to import said class
    and construct an instance of it using the parameters. This makes
    configuring objects as JSON dictionaries convenient and avoids
    unnecessary imports.

    example: 'trainers.BaseTrainer' or {'name': 'trainers.BaseTrainer', 'parameters': {...}}

    :param config: string of class name or dict with class name and kwargs.
    :return: instance of specified class.
    """
    if isinstance(config, str):
        config = {'name': config, 'parameters': {}}

    import_str = config['name']
    parts = import_str.split('.')
    module_str = '.'.join(parts[:-1])
    class_str = parts[-1]

    try:
        module = importlib.import_module(module_str)
    except ImportError:
        raise ImportError('Cannot import module %s when trying to load %s' %
                          (module_str, import_str))

    try:
        class_obj = getattr(module, class_str)
    except AttributeError:
        raise AttributeError('Found no attribute %s in %s when trying to load %s' %
                             (class_str, module_str, import_str))

    kwargs = config['parameters']
    obj = class_obj(**kwargs)

    return obj


def log_config(config):
    """Log config dictionary to MLFlow."""
    def iter_dict(d, prefix=None):
        """Iterate a dictionary recursively."""
        for key, value in d.items():
            new_prefix = key if prefix is None else prefix + '_' + key
            if isinstance(value, dict):
                yield from iter_dict(value, new_prefix)
            else:
                yield new_prefix, value

    for name, param in iter_dict(config['dataset'], 'dataset'):
        name = name.replace('_name', '').replace('_parameters', '')
        mlflow.log_param(name, param)

    for name, param in iter_dict(config['model'], 'model'):
        name = name.replace('_name', '').replace('_parameters', '')
        mlflow.log_param(name, param)

    for name, param in iter_dict(config['trainer'], 'trainer'):
        name = name.replace('_name', '').replace('_parameters', '')
        mlflow.log_param(name, param)


def read_config(config_path):
    """Read config JSON and expand class name strings."""
    with open(config_path, mode='rt') as f:
        config = json.load(f)

    def iter_dict(d):
        """Iterate a dictionary recursively."""
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'name':
                    continue
                else:
                    d[key] = iter_dict(value)
        elif isinstance(d, list):
            for i, item in enumerate(d):
                d[i] = iter_dict(item)
        elif isinstance(d, str) and '.' in d:
            return {'name': d, 'parameters': {}}
        return d

    iter_dict(config)

    return config


def is_debugging():
    """Return true if debugger is found in stack trace."""
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True

    return False


def build_tmp_dir():
    """Create a temporary dir that is deleted on termination."""
    tmp_path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tmp_path)

    return tmp_path
