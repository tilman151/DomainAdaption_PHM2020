"""Metrics for domain adaption."""

import torch
import torch.nn as nn

from metrics import BaseMetric


class RMSE(BaseMetric):
    def __init__(self, device):
        super(RMSE, self).__init__('root_mean_squared_error', device)

    def _evaluate(self, model, data):
        mse = torch.zeros(())
        elements = len(data.dataset)
        criterion = nn.MSELoss(reduction='none')
        for _, _, inputs, targets in data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = model(inputs)

            mse += criterion(outputs.squeeze(), targets).sum() / elements

        rmse = torch.sqrt(mse).item()
        results = {'scalars': {self.name: rmse}}

        return rmse, results


class RULScore(BaseMetric):
    def __init__(self, device):
        super(RULScore, self).__init__('rul_score', device)

    def _evaluate(self, model, data):
        score = 0
        elements = len(data.dataset)
        for _, _, inputs, targets in data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = model(inputs)

            dist = outputs.squeeze() - targets
            for i, d in enumerate(dist):
                dist[i] = (- d / 13) if d < 0 else (d / 10)
            dist = torch.exp(dist) - 1
            score += dist.sum().item() / elements

        results = {'scalars': {self.name: score}}

        return score, results
