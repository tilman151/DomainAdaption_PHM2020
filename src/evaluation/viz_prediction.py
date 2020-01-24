import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

import datasets.cmapss as data


@torch.no_grad()
def visualize_predictions(model_path, fd, window_size, normalization='minmax'):
    dataset = data.cmapss(fd, window_size, batch_size=16, shuffle=False, normalization=normalization)
    model = torch.load(model_path, map_location='cpu')
    model = model.to('cpu')
    model.eval()

    predictions = []
    for features, _ in dataset.training(num_workers=0):
        outputs = model(features.to('cpu'))
        predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = dataset.train_data.targets
    split_idx = np.argwhere(targets == 1).squeeze() + 1
    x_vals = np.split(np.arange(targets.shape[0]), split_idx)
    predictions = np.split(predictions, split_idx)
    targets = np.split(targets, split_idx)

    plt.figure(figsize=(10, 10))
    plt.xlabel('Cycles')
    plt.ylabel('RUL')
    cmp = plt.get_cmap('tab10')

    for x, t, p in zip(x_vals, targets, predictions):
        plt.plot(x, t, color=cmp.colors[0])
        plt.plot(x, p, color=cmp.colors[1])
        break
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of RUL prediction')
    parser.add_argument('model_path', help='Path to model file.')
    parser.add_argument('--fd', required=True, type=int, help='Number of CMAPSS partition.')
    parser.add_argument('--window_size', required=True, type=int, help='Size of window for dataset.')
    parser.add_argument('--norm', required=True, help='Type of normalization to use in data.')
    opt = parser.parse_args()

    visualize_predictions(opt.model_path, opt.fd, opt.window_size, opt.norm)
