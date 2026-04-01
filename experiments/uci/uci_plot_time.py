import pandas as pd
import ast  # For literal_eval
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast

# Times new roman font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
# Bolds font
plt.rcParams.update({'font.size': 14})
# legend fontsize
plt.rcParams["legend.fontsize"] = 12

def read_results_csv(file_path):
    results = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert the 'Results' column from string to dictionary
            row['Results'] = ast.literal_eval(row['Results'])
            # Only keep 'Method' and 'Results' columns
            results[row['RegressionMethod']] = row['Results']

    return results

def plot_metrics(dataset):
    # Read the CSV file into a Pandas DataFrame
    csv_file = 'experiments/uci/results/metrics/' + dataset + '_time.csv'
    results = read_results_csv(csv_file)

    methods = ['CV_100', 'CV_1000', 'VIPS']
    # order the methods
    run_time = []
    # RMSEs = []
    # NLPDs = []
    N_totals = []
    M_batch = []

    run_time = [np.cumsum(results[method]['time']) for method in methods]
    N_totals = [np.cumsum(results[method]['N']) for method in methods]
    M_batch = [results[method]['M'] for method in methods]

    for method in methods:
        print(method)
        print("Time", np.cumsum(results[method]['time'])[-1])
        print("NLPD", results[method]['NLPD'][-1])
        print("RMSE", results[method]['RMSE'][-1])
        print("M", results[method]['M'][-1])

    # # Create diccionary with different colors for each method
    # dark blue, orange, green, red, purple
    colors = {'CV_100': '#00008b', 'VIPS': '#800080', 'CV_1000': '#6495ed', 'Gradient': '#7fb3d5'}
    labels = {'CV_100': 'Oracle', 'VIPS': 'VIPS', 'CV_1000': 'Heuristic', 'Gradient': 'Bui et al.'}

    plt.figure(figsize=(6, 2.6))
    batches = range(1, len(run_time[0])+1)
    for i, method in enumerate(methods):
        plt.plot(batches, run_time[i], label=labels[method], color=colors[method])
        if method == 'VIPS':
            for batch, M in enumerate(M_batch[i], start=1):
                if batch % 5 == 0 or batch == 1:
                    plt.text(batch-1.5, max(run_time[2])+700, '$\mathrm{M}_{\mathrm{vips}} = $', fontsize=9, color='k')
                    plt.text(batch+1, max(run_time[2])+700, str(M), fontsize=11, color=colors[method])
    plt.xlabel('Batch Number')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1, fontsize=11)
    plt.grid(axis='x')
    plt.xticks([1, 5, 10, 15, 20])

    plt.tight_layout()
    plt.savefig('experiments/uci/results/plots/' + dataset + '_plot_all.pdf')


if __name__ == "__main__":

    # Pass the name of the dataset as an argument
    import argparse
    parser = argparse.ArgumentParser(description='Plot metrics for UCI datasets')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    args = parser.parse_args()
    dataset = args.dataset
    plot_metrics(dataset)