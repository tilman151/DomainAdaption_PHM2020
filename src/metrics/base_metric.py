"""Base class for evaluation metrics."""
import os
import torch
import torch.nn as nn


class BaseMetric:
    def __init__(self, name, device):
        self.name = name
        self.device = device

    def run(self, model, data):
        if isinstance(model, str) and os.path.exists(model):
            model = torch.load(model, map_location='cpu')
        assert isinstance(model, nn.Module), 'Model object is not a nn.Module or a file to load it from.'

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            metric, result_dict = self._evaluate(model, data)

        return metric, result_dict

    def __str__(self):
        return self.name

    def _evaluate(self, model, data):
        raise NotImplementedError()
