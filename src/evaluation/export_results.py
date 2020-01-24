"""Exports all RUL experiments to a data frame."""

import argparse
import mlflow
import re
import pandas as pd
import numpy as np

from evaluation.rul_experiments import evaluate_rul_experiment, get_best_value


def export_all(mlflow_uri):
    client = mlflow.tracking.MlflowClient(mlflow_uri)
    experiments = client.list_experiments()

    df = export_transfer(experiments, mlflow_uri)
    df.to_csv('transfer.csv')

    df = export_baseline(client, experiments)
    df.to_csv('baseline.csv')


def export_baseline(client, experiments):
    regex = re.compile('cmapss_.{3,5}_baseline$')
    baseline_experiments = [e for e in experiments if regex.match(e.name) is not None]
    print('Found %d baseline experiments...' % len(baseline_experiments))
    df = pd.DataFrame()
    for e in baseline_experiments:
        runs = client.search_runs(experiment_ids=e.experiment_id)
        replications = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]

        run_df = pd.DataFrame(np.zeros((len(replications), 8)),
                              columns=['rul_score_1', 'rul_score_2', 'rul_score_3', 'rul_score_4',
                                       'mse_1', 'mse_2', 'mse_3', 'mse_4'],
                              index=[e.name] * len(replications))
        for i, run in enumerate(replications):
            best_scores = [get_best_value(client, 'evaluation_rul_score_%d' % i, run) for i in range(1, 5)]
            best_rmse = [get_best_value(client, 'evaluation_rmse_%d' % i, run) for i in range(1, 5)]
            run_df.iloc[i] = best_scores + best_rmse
        df = df.append(run_df)
    return df


def export_transfer(experiments, mlflow_uri):
    regex = re.compile('cmapss_.+2.+_.{3,5}$')
    transfer_experiments = [e for e in experiments if regex.match(e.name) is not None]
    print('Found %d transfer experiments...' % len(transfer_experiments))
    df = evaluate_rul_experiment(mlflow_uri, transfer_experiments[0].name)
    for e in transfer_experiments[1:]:
        df = df.append(evaluate_rul_experiment(mlflow_uri, e.name))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exports all experiments to CSV')
    parser.add_argument('mlflow_uri', help='URI for MLFlow Server.')
    opt = parser.parse_args()

    export_all(opt.mlflow_uri)

