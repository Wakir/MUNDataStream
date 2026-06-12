## MUNDataStream

This repository contains a code and results for experiments conducted in paper "**Unlearning-based sliding window for continual learning under concept drift**".

### Dependecies and setup

This work is done using Python 3.8.18 and stream_learn 0.8.21. Additional dependencies can be found in `requirements.txt` file. To setup the repository you need to simple run a command in existing Python enviroment:

```
pip install -r requirments.txt
```

### Running the Experiments

Running the experiment is done by calling function `experiment_sud.py`,  `experiment_grad.py`, `experiment_incr.py` or `experiment_sem.py` with the desired input parameters. 

`Experiment_sud.py` contains experiment for sudden drift datastreams (MNIST and CIFAR-10).
`Experiment_grad.py` contains experiment for gradual drift datastreams (MNIST and CIFAR-10).
`Experiment_incr.py` contains experiment for incremental drift datastreams (MNIST and CIFAR-10).
`Experiment_incr.py` contains experiment for semantic drift datastreams (F-MNIST and CIFAR-10).

The hiperparameters uses in following experiments are:

* `chunk_sizes` - number of samples in each data batch.
* `noise_precents` - specifies the fraction of all pixel values in each image that are randomly selected and perturbed by adding Gaussian noise before concept drift ([0, 1] for sudden, gradual and incremental drifts).
* `delta_noise` - specifies the fraction of all pixel values in each image that are randomly selected and perturbed by adding Gaussian noise after concept drift ([0, 1] for sudden, gradual and incremental drifts).
* 'overlap_chunks' - number of overlap chunks (for gradual drift)
* 'noise_steps' - linear change of noise between each chunk during drift change (for incremental drift)
* 'semantic_cases_1' - type of semantic case before drift (for semantic drift)
* 'semantic_cases_2' - type of semantic case after drift (for semantic drift)
* 'dataset_name' - dataset name used as a base for datastream (CIFAR-10, MNIST for sudden,gradual and incremental streams, CIFAR-10 and FASHION-MNIST for semantic)
* `algorithms` - alghoritms used in experiments (Sliding for SW, Unlearning for UIL)
* `random_seeds` - the seeds used by the random number generator.
* `window_sizes` - number of maximum chunks in sliding window
* `learning_rates` - value of learning rate in machine learning model
* `unlearning_rates` - strenght of unlearning, applied to an exit value of unlearning operation
* `metrics` - list of metric functions or single metric function.

### Results

Achived results are categoriased based on stream type (`Synthetic` or `INSECTS`). In the following subdirectories:
* `results` - contains mlflow files with full experiment results.
* `figures` - contains excel tables and plots with avareage results of the following experiments.
