"""Script for retrieving the eval matrix of a RUL experiment."""

import argparse
import mlflow
import numpy as np
import pandas as pd


def evaluate_rul_experiment(mlflow_uri, experiment_name):
    """Retrieve and evaluate RUL experiment."""
    client = mlflow.tracking.MlflowClient(mlflow_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    print('Evaluate experiment "%s" with id %s' % (experiment.name, experiment.experiment_id))

    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    top_level_runs = [r for r in runs if 'mlflow.parentRunId' not in r.data.tags]
    replications = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
    print('Found %d top-level runs...' % len(top_level_runs))
    _, counts = np.unique([r.data.tags['mlflow.parentRunId'] for r in replications], return_counts=True)
    if counts.sum() == (counts[0] * counts.shape[0]):
        print('Found %d replications each...' % counts[0])
    else:
        raise ValueError('Heterogeneous number of replications found...')

    replications = sorted(replications, key=matrix_key)

    df = pd.DataFrame(np.zeros((len(replications), 4)),
                      columns=['percent_broken', 'percent_fail_runs', 'rul_score', 'mse'])
    for i, run in enumerate(replications):
        best_scores = get_best_value(client, 'evaluation_rul_score', run)
        best_rmse = get_best_value(client, 'evaluation_root_mean_squared_error', run)
        percent_broken = run.data.params['dataset_target_dataset_percent_broken']
        percent_fail_runs = run.data.params['dataset_target_dataset_percent_fail_runs']
        df.iloc[i] = [percent_broken, percent_fail_runs, best_scores, best_rmse]

    df.index = [experiment_name] * len(replications)
    print('Return %d runs...' % len(df))

    return df


def matrix_key(run):
    """Sort key for matrix."""
    return float(run.data.params['dataset_target_dataset_percent_broken']) * 100 - \
           float(run.data.params['dataset_target_dataset_percent_fail_runs'])


def get_best_value(client, metric, run):
    """Return best value of selected metric."""
    metric_history = client.get_metric_history(run.info.run_id, metric)
    best_value = min(metric_history, key=lambda x: x.value)
    return best_value.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RUL Experiment.')
    parser.add_argument('mlflow_uri', help='URI to MLFlow tracking server')
    parser.add_argument('-e', '--experiment', help='Name of experiment')
    opt = parser.parse_args()

    data = evaluate_rul_experiment(opt.mlflow_uri, opt.experiment)
    data.to_csv('%s.csv' % opt.experiment)
