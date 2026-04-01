import argparse
import csv
import json
import os
import yaml

import numpy as np
import tensorflow as tf
import utils_uci as utils

# Create a dictionary mapping method names to their respective functions
method_functions = {
    'CV': utils.run_CV,
    'OIPS': utils.run_OIPS,
    'VIPS': utils.run_VIPS,
    'Full_GP': utils.Full_GP,
    'CV_100': utils.run_CV,
    'CV_1000': utils.run_CV
}

def extract_method_info(config, method_name):
    methods = config.get('methods', [])

    for method in methods:
        if method.get('Name') == method_name:
            return method.get('method_parameters', {})

    return None

def calculate_mean_metrics(file_path):
    aggregated_data = {}
    mean_metrics = {}
    std_metrics = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            all_metrics = json.loads(row['Metrics'].replace("'", "\""))

            for key, values in all_metrics.items():
                if key not in aggregated_data:
                    aggregated_data[key] = [[] for _ in range(len(values))]
                
                for i, value in enumerate(values):
                    aggregated_data[key][i].append(value)

    # Calculate the mean for each batch of each metric
    for key in aggregated_data:
        mean_metrics[key] = [np.mean(values) for values in aggregated_data[key]]

    # Calculate the standard deviation for each batch of RMSE and NLPD, and M
    std_metrics['RMSE'] = [np.std(values) for values in aggregated_data['RMSE']]
    std_metrics['NLPD'] = [np.std(values) for values in aggregated_data['NLPD']]

    # Convert M and N to integers
    if 'M' in mean_metrics:
        mean_metrics['M'] = [int(M) for M in mean_metrics['M']]
        std_metrics['M'] = [np.std(values) for values in aggregated_data['M']]

    mean_metrics['N'] = [np.round(N,0) for N in mean_metrics['N']]

    return mean_metrics, std_metrics

def run_method(method_name, method_function, train_data, test_data, no_batches, test_per_batch, method_params):
    X_train, y_train = train_data
    X_test, y_test = test_data

    if method_name == 'Full_GP':
        return method_function(X_train, y_train, no_batches, X_test, y_test, test_per_batch)
    else:
        return method_function(X_train, y_train, no_batches, X_test, y_test, test_per_batch, **method_params)

def write_results_to_csv(file_path, method_name, reg_params, method_params, mean_metrics, std_metrics = None):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['RegressionMethod', 'RegressionParameters', 'MethodParameters', 'Results', 'Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({
            'RegressionMethod': method_name,
            'RegressionParameters': str(reg_params),
            'MethodParameters': str(method_params),
            'Results': mean_metrics,
            'Std': std_metrics
        })

def run_fold(method_name, train_data, test_data, no_batches, test_per_batch, method_params):
    metrics = run_method(method_name, method_functions[method_name], train_data, test_data, no_batches, test_per_batch, method_params).run()
    return metrics

def run_regression(config, method_name, dataset_name):
    reg_params = config['regression_parameters']
    dataset = config['dataset']
    method_params = extract_method_info(config, method_name)

    no_batches = reg_params['no_batches']
    k_folds = reg_params['n_k_folds']
    test_per_batch = reg_params['test_per_batch']
    order = reg_params['order']

    all_train_data, all_test_data = utils.load_data(dataset, n_k_folds=k_folds, seed=reg_params['seed'], 
                                                    no_batches=no_batches, order=order)
    
    if k_folds is not None:
        temp_file_path = f'experiments/uci/results/metrics/{dataset}_{method_name}.tmp'
        with open(temp_file_path, 'a', newline='') as csvfile:
            fieldnames = ['Fold', 'Metrics']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(k_folds):
                train_data, test_data = all_train_data[i], all_test_data[i]
                metrics = run_fold(method_name, train_data, test_data, no_batches, test_per_batch, method_params)
                writer.writerow({'Fold': f"Fold {i}", 'Metrics': metrics})
                csvfile.flush()
                tf.keras.backend.clear_session()  
                
        metrics, std = calculate_mean_metrics(temp_file_path)
        os.remove(temp_file_path)
    else:
        train_data, test_data = all_train_data[0], all_test_data[0]
        metrics = run_fold(method_name, train_data, test_data, no_batches, test_per_batch, method_params)
        std = None
        tf.keras.backend.clear_session() 

    results_file_path = f'experiments/uci/results/metrics/{dataset_name}.csv'
    write_results_to_csv(results_file_path, method_name, reg_params, method_params, metrics, std)

    print(f"Finished running {method_name} on {dataset} dataset.")
    print(f"Results saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression on UCI datasets')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--method', type=str, help='Name of the regression method')
    args = parser.parse_args()

    configs_dir = 'experiments/uci/configs/'

    dataset = args.dataset
    yaml_file_path = os.path.join(configs_dir, f'{dataset}.yaml')

    try:
        with open(yaml_file_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Config file for {dataset} not found.")
        exit(1)

    method = args.method
    run_regression(config, method, dataset)
    