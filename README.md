This repository contains a code and results for experiments conducted in paper "**Unlearning-based sliding window for continual learning under concept drif**".

### Dependecies and setup

This work is done using Python 3.8.18 and stream_learn 0.8.21. Additional dependencies can be found in `requirements.txt` file. To setup the repository you need to simple run a command in existing Python enviroment:

```
pip install -r requirments.txt
```

### Running the Experiments

Running the experiment is done by calling function `src/experiment1.py` or `src/experiment2.py` with the desired input parameters. 

`Experiment1.py` contains experiment for synthetic datasets generated from `StreamGenerator`.
`Experiment2.py` uses INSECTS dataset located in `src/real_str`.

The hiperparameters uses in following experiments are:

* `random_seeds` - The seeds used by the random number generator.
* `imbalance` - Procentage of minority class in synthetic stream.
* `scales` - tested sizes of the bootstrapped subsets concerning the original training set.
* `n_classifiers` - the number of classifiers generated with one data chunk.
* `base_classifier` - classifier type used in IMB-WAE and other reference methods.
* `quality_measure` - metric used in pruning.
* `reference_methods` - List of all reference methods used in experiment.
* `metrics` - List of metric functions or single metric function.

### Results

Achived results are categoriased based on stream type (`Synthetic` or `INSECTS`). In the following subdirectories:
* `full` - contains numpy files with full experiment results.
* `tables` - contains excel tables with avareage results of the following experiments for all chunks in stream.
* `plots` - contains comparison plots of proposed method and reference methods for each chunk in stream.
