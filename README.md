# Evaluation of Unsupervised Domain Adaption for RUL Estimation

This repository contains the code and raw results for the upcoming ICPHM2020 paper 
`A Novel Evaluation Framework for Unsupervised Domain Adaption on Remaining Useful Lifetime Estimation`.

Unsupervised Domain Adaption (DA) is an approach for adapting a data-driven model to new data without labels.
Recent work on Remaining Useful Lifetime (RUL) estimation of aero engines yielded promising results for this approach.
However, the current evaluation framework for DA is of limited significance when used for RUL estimation.
It assumes a use case where a large number of fully degraded systems are available for adaption, which makes unsupervised DA in itself unnecessary.
It is shown that the current framework overestimates adaption performance and obscures potential, negative effects of DA on performance.
We propose a novel evaluation framework for unsupervised DA, specialized in RUL estimation, that takes the number of available systems and their grade of degradation into account.
It enables an informed performance comparison of DA methods.

The results show that the grade of degradation is the determining factor for performance, while the number of systems has next to no influence. This repository is able to reproduce the results of the paper and enable evaluating your own algorithms with this framework.

Feel free to contact me, if you have questions about this work.

## Installation

The code was implemented for and tested with Python 3.7. Following additional packages are required:

    pytorch=1.4
    mlflow=1.5
    tensorboard=1.14
    matplotlib=3.1
    scikit-learn=0.22
    pandas
    statmodels
    
Optionally you can also use the provided conda environment file to recreate the environment.

Create the folders `data\CMAPSS` in the root directory of the repository.
Place the files of the CMAPSS dataset in it.

## Basic Usage
### Manual Experiments

Experiments are described by JSON files.
To start a simple experiment, call `run.py` with the desired configuration JSON file:

    python run.py -e 125 -r 5 -d cpu -s 42 -w 0 configs/cmapss/three2one/cmapss_three2one_jan.json
    
* `-e`: the number of training epochs
* `-r`: the number of replications with seperate seeds
* `-d`: the training device (e.g. cuda:0 for GPU training)
* `-s`: the random seed
* `-w`: the number of worker processes for loading data (0 recommended)

The results are logged into the folder `mlruns` and can be inspected with the `mlflow ui` tool.
Additionally there is a TensorBoard event file written to the folder `results`.

### Reproducing Experiments

If you want to reproduce all or part of the experiments of the paper, please use the shell scripts in the
folder `run_scrips`.
They use the proper random seed and run deterministically.

## Inspecting Results
### Paper Results

To investigate the raw results of the papers experiments, load the CSVs in the `results` folder:

    pandas.read_csv(transfer.csv, index_col=0)
    pandas.read_csv(baseline.csv, index_col=0)
    
The plots of the paper can be reproduced by the script `src/plots/plot_results.py`.
It additionally prints the aggregated result tables as renderable LaTeX code to the console.

### Own Results

The `src/evaluation/evaluate_baseline.py` will evaluate all baseline experiments found in mlflow
on all subsets of CMAPSS.
The results are written back to mlflow.

If you want to export your own results from mlflow, use the script `src\evaluation\export_results.py`.
It will export properly named transfer and baseline experiments to CSV.
