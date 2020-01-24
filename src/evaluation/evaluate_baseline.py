import torch
import os
import mlflow
import re

import utils
import metrics.domain_adaption as metrics
import datasets.cmapss as data
import datasets.domain_adaption as da


client = mlflow.tracking.MlflowClient('http://127.0.0.1:5000')

experiments = client.list_experiments()
pattern = re.compile('cmapss_[a-z]+_baseline$')
experiments = [e for e in experiments if pattern.match(e.name) is not None]
print('Found experiments %d' % len(experiments))

fd1 = data.cmapss(1, 30)
fd2 = data.cmapss(2, 30)
fd3 = data.cmapss(3, 30)
fd4 = data.cmapss(4, 30)

fd1 = da.deep_adaption_dataset(fd1, fd1, batch_size=512)
fd2 = da.deep_adaption_dataset(fd2, fd2, batch_size=512)
fd3 = da.deep_adaption_dataset(fd3, fd3, batch_size=512)
fd4 = da.deep_adaption_dataset(fd4, fd4, batch_size=512)

temp_path = utils.build_tmp_dir()
rul_score = metrics.RULScore(device='cpu')
rmse = metrics.RMSE(device='cpu')

for e in experiments:
    runs = client.search_runs(e.experiment_id)
    runs = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]

    for r in runs:
        print('Process run %s' % r.info.run_id)
        models = client.list_artifacts(r.info.run_id, 'models')
        models = sorted(models, key=lambda x: int(x.path.split('_')[1][:-4]))
        for i, m in enumerate(models, start=1):
            artifact_path = os.path.join(temp_path, r.info.run_id)
            os.makedirs(artifact_path, exist_ok=True)
            model_path = client.download_artifacts(r.info.run_id, m.path, artifact_path)
            model = torch.load(model_path)
            model = model.to('cpu')
            model.eval()
            for j, dataset in enumerate([fd1, fd2, fd3, fd4], start=1):
                res = rul_score.run(model, dataset.evaluation(num_workers=0))
                client.log_metric(r.info.run_id, 'evaluation_rul_score_%d' % j, res[0], step=i)
                res = rmse.run(model, dataset.evaluation(num_workers=0))
                client.log_metric(r.info.run_id, 'evaluation_rmse_%d' % j, res[0], step=i)
