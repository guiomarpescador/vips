Adjusting Model Size in Continual Gaussian Processes: How Big is Big Enough?
===========================================================
Guiomar Pescador-Barrios, Sarah Filippi and Mark van der Wilk.

Accepted (Spotlight) at ICML 2025. 

## Requirements
Create a virtual environment for python >=3.10 using your favourite tool. Install packages and requirements doing ```pip install -r requirements.txt```.

## Structure
- Model and inducing points selection algorithms are in ```src/```
- Experiments are in ```experiments/```

## Experiments

###  Data Types 
Run the following command to reproduce the experiments for the toy dataset:
```python3 experiments/data_types/run_experiment.py```

### Synthetic data

Run the following command to reproduce the experiments for the toy dataset:
```python3 experiments/toy/run_toy.py```
and for plotting the results:
```python3 experiments/toy/plot_toy.py```

Results and plots will be saved in ```experiments/toy/results/```.

### Timing experiments
Run the following command to reproduce the experiments for the timing experiments:
```python3 experiments/uci/uci_metrics.py --dataset naval_time --method {method}```,
where ```{method}``` is the inducing points selection method to use.

Possible values for methods are:
- CV_100 (referring to greedy variance in  with 100 inducing points)
- CV_1000 (referring to greedy variance in with 1000 inducing points)
- VIPS (Ours)

To plot the results, run:
```python3 experiments/uci/uci_plot_time.py --dataset naval```

### UCI datasets
Run the following command to reproduce the experiments for the UCI datasets:
```python3 experiments/uci/uci_metrics.py --dataset {dataset} --method {method}```,

where ```{dataset}``` is the name of the dataset and ```{method}``` is the inducing points selection method to use. 

Possible values for methods are:
- CV (referring to Conditional Variance)
- OIPS (Galy-Fajou & Opper, 2021)
- VIPS (Ours)
- Gradient (Bui et al., 2017)

The configuration for each data set should be set in ```experiments/uci/configs.py```. The following datasets are available:
- concrete (1030,8)
- skillcraft (3338,19)
- kin8nm (7372, 8)
- naval (11934, 16)
- elevators (16599, 18)
- bike (17379, 17)
- naval (11934, 14)

To run multiple hyperparameters for the UCI datasets, run:
```python3 experiments/uci/uci_pareto.py --datasets {dataset1} {dataset2} ...```
and plot results using
```python3 experiments/uci/uci_plot_pareto.py --datasets {dataset1} {dataset2} ...```


### Magnetometer
Run the following command to reproduce the first experiment for the magnetometer dataset:
```python3 experiments/magnetometer/run_experiment_1.py```

Run the following command to reproduce the second experiment for the magnetometer dataset:
```python3 experiments/magnetometer/run_experiment_2.py```

For plotting the results, run:
```python3 experiments/magnetometer/plot_experiment_1.py```
```python3 experiments/magnetometer/plot_experiment_2.py```