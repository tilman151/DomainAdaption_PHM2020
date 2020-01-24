import matplotlib.pyplot as plt
import numpy as np

import datasets.cmapss as cmapss


def plot_features(norm):
    plt.figure(figsize=(15, 10))
    for i, percent in enumerate([1.0, 0.8, 0.6, 0.4, 0.2], start=1):
        ax1 = plt.subplot(5, 1, i)
        dataset = cmapss.cmapss(1, 1, shuffle=False, percent_fail_runs=percent,
                                percent_broken=None, normalization=norm)
        train_features = dataset.train_data.features
        eval_features = dataset.eval_data.features
        x_steps = np.arange(1.5 * train_features.shape[1], step=1.5)
        ax1.violinplot(np.squeeze(train_features), x_steps + 0.75, showmeans=True)
        ax1.violinplot(np.squeeze(eval_features), x_steps + 1.25, showmeans=True)
        ax2 = ax1.twinx()
        violin = ax2.violinplot(dataset.train_data.targets, [1.5 * train_features.shape[1] + 1], showmeans=True)
        plt.setp(violin['bodies'], facecolor='red')
        plt.setp(violin['cmins'], edgecolor='red')
        plt.setp(violin['cmaxes'], edgecolor='red')
        plt.setp(violin['cbars'], edgecolor='red')
        plt.setp(violin['cmeans'], edgecolor='red')
        plt.xticks(x_steps + 1, np.arange(train_features.shape[1]) + 1)
        ax1.set_ylim(-1.5, 1.5)
        plt.title('%d%% Runs to Failure' % int(percent * 100))

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(15, 10))
    for i, percent in enumerate([0.0, 0.2, 0.4, 0.6, 0.8], start=1):
        ax1 = plt.subplot(5, 1, i)
        dataset = cmapss.cmapss(1, 1, shuffle=False, percent_fail_runs=None,
                                percent_broken=percent, normalization=norm)
        train_features = dataset.train_data.features
        eval_features = dataset.eval_data.features
        x_steps = np.arange(1.5 * train_features.shape[1], step=1.5)
        ax1.violinplot(np.squeeze(train_features), x_steps + 0.75, showmeans=True)
        ax1.violinplot(np.squeeze(eval_features), x_steps + 1.25, showmeans=True)
        ax2 = ax1.twinx()
        violin = ax2.violinplot(dataset.train_data.targets, [1.5 * train_features.shape[1] + 1], showmeans=True)
        plt.setp(violin['bodies'], facecolor='red')
        plt.setp(violin['cmins'], edgecolor='red')
        plt.setp(violin['cmaxes'], edgecolor='red')
        plt.setp(violin['cbars'], edgecolor='red')
        plt.setp(violin['cmeans'], edgecolor='red')
        plt.xticks(x_steps + 1, np.arange(train_features.shape[1]) + 1)
        ax1.set_ylim(-1.5, 1.5)
        ax2.set_ylim(-5, 130)
        plt.title('%d%% Truncated Steps before Failure' % int(percent * 100))

    plt.tight_layout()
    plt.show()


def plot_features_z():
    plot_features('z')


def plot_features_minmax():
    plot_features('minmax')


def plot_features_robust():
    plot_features('robust')


if __name__ == '__main__':
    plot_features_minmax()
