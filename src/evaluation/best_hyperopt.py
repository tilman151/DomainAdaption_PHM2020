import argparse
import mlflow
import pprint


def get_best_hyperparams(mlflow_uri, experiment, metric):
    client = mlflow.tracking.MlflowClient(mlflow_uri)

    experiment = client.get_experiment_by_name(experiment)
    print('Evaluate experiment "%s" with id %s' % (experiment.name, experiment.experiment_id))

    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    print('Found %d runs...' % len(runs))

    best_run = (None, 1e10, [])
    for r in runs:
        if'mlflow.parentRunId' in r.data.tags:
            continue
        nested_runs = client.search_runs(r.info.experiment_id,
                                         filter_string='tags.mlflow.parentRunId = "%s"' % r.info.run_id)
        if nested_runs:
            best_value = sum(get_best_value(client, metric, nested) for nested in nested_runs) / len(nested_runs)
        else:
            best_value = get_best_value(client, metric, r)
        if best_value < best_run[1]:
            best_run = (r, best_value, nested_runs)

    print('Best run has id %s with %s of %.3f' % (best_run[0].info.run_id, metric, best_run[1]))
    if best_run[2]:
        nested_run_ids = tuple(r.info.run_id for r in best_run[2])
        print('\tas the mean of runs' + ', %s' * len(nested_run_ids) % nested_run_ids)
    pprint.pprint(best_run[0].data.params)


def get_best_value(client, metric, run):
    metric_history = client.get_metric_history(run.info.run_id, metric)
    best_value = min(metric_history, key=lambda x: x.value)
    return best_value.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get best hyperparameters from vary.py experiment.')
    parser.add_argument('mlflow_uri', help='URI to MLFlow server.')
    parser.add_argument('-e', '--experiment', required=True, help='Name of experiment to evaluate.')
    parser.add_argument('-m', '--metric', required=True, help='Name of metric to evaluate.')
    opt = parser.parse_args()

    get_best_hyperparams(opt.mlflow_uri, opt.experiment, opt.metric)
