
import json
import numpy as np
import utils_regression as utils

file_path = 'experiments/toy/results/results.json' 

def metrics_to_dict(metrics):
    return {'RMSE': list(metrics[0]), 'NLPD': list(metrics[1]), 'M': list(metrics[2])}

def run_experiment():

    no_batches = 4
    
    np.random.seed(42)
    X_train = np.random.rand(1000, 1) * 12
    y_train = np.sin(2*X_train) + np.cos(5*X_train) + np.random.randn(1000, 1) 
    # Create test set on grid
    X_test = np.linspace(0, 10, 500).reshape(-1, 1)
    y_test = np.sin(2*X_test) + np.cos(5*X_test)

    Ms = [10, 20, 30]

    # List to store all the results
    results = []

    for M in Ms:
        metrics = utils.run_regression(X_train, y_train, no_batches, 'MDPP', (X_test, y_test), M=M, eta=1e-20)
        results.append({'Method': f'CV_{M}', 'Metrics': metrics_to_dict(metrics)})

    metrics = utils.run_regression(X_train, y_train, no_batches, 'VIPS', (X_test, y_test))
    results.append({'Method': 'VIPS', 'Metrics': metrics_to_dict(metrics)}) 

    metrics = utils.run_exact(X_train, y_train, X_test, y_test, no_batches)
    results.append({'Method': 'Full_GP', 'Metrics': metrics_to_dict(metrics)})

    # Save the results to a JSON file
    with open(file_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)


if __name__ == "__main__":
    run_experiment()