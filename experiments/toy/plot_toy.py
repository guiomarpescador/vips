import json
from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
# Line width
plt.rcParams['lines.linewidth'] = 2.0
# legend fontsize
plt.rcParams["legend.fontsize"] = 10

file_path = 'experiments/toy/results/results.json'  # Replace with your actual file path

def load_results_from_json(file_path):
    with open(file_path, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data

def plot_results():
    results = load_results_from_json(file_path)

    batches = range(1, len(results[0]['Metrics']['RMSE'])+1) 
    colors = {'CV_10':  '#00008b', 'CV_20': '#6495ed', 'CV_30': '#7fb3d5', 'Full_GP': '#000000', 'VIPS' :'#800080'}
    markers = {'CV_10': 'x', 'CV_20': 'x', 'CV_30': 'x', 'Full_GP': 'x', 'VIPS': 'x'}
    linestyle = {'CV_10': '-', 'CV_20': '-', 'CV_30': '-', 'CV_15': '--', 'VIPS': '--'}
    labels = {'CV_10': 'M = 10', 'CV_20': 'M = 20', 'CV_30': 'M = 30', 'Full_GP': 'Exact GP', 'VIPS': 'VIPS'}

    plt.figure(figsize=(6, 2.5))

    for result in results:
        method = result['Method']
        # Round up to 1 decimal
        metrics = result['Metrics']
        if method == 'Full_GP':
            full_gp = metrics['RMSE']
            continue
        plt.plot(batches, metrics['RMSE'], label=labels[method], marker=markers[method], color=colors[method], linestyle=linestyle[method])


    for batch in batches:
        plt.hlines(y=full_gp[batch-1], xmin=batch-1, xmax=batch, color='k', linestyle='--', alpha=0.5, label='Exact GP' if batch == 4 else "")
        if batch == batches[-1]:
            plt.vlines(x=batch, ymin=full_gp[batch-1], ymax=full_gp[batch-1], color='k', linestyle='--', alpha=0.5)
            break
        plt.vlines(x=batch, ymin=full_gp[batch], ymax=full_gp[batch-1], color='k', linestyle='--', alpha=0.5)


    plt.xlim(xmin=0.8)
    plt.xticks(batches)
    # Annotations and labels
    plt.xlabel('Batch Number')
    plt.ylabel('RMSE')
    # Size of the legend
    plt.tight_layout()

    # Make space for the legend
    plt.legend(loc='upper center', bbox_to_anchor=(1.22, 1), ncol=1, fontsize=11)
    
    for result in results:
        method = result['Method']
        metrics = result['Metrics']


        if method == 'VIPS':
            for batch, M in enumerate(metrics['M'], start=1):
                horizontal_offset = -0.2 
                number_offset = 0.45
                # Annotate the 'M =' part above the graph
                plt.text(batch + horizontal_offset, metrics['RMSE'][0] + 0.08, 
                        '$\mathrm{M}_{\mathrm{vips}} = $', fontsize=9, va='center')

                # Annotate the number part in color above the graph
                plt.text(batch + horizontal_offset + number_offset, metrics['RMSE'][0] + 0.08, 
                        f'{int(M)}',
                        fontsize=11, va='center', color=colors[method])

    plt.grid(axis='x')
    plt.tight_layout()
    # Save figure
    plt.savefig('experiments/toy/results/toy.pdf', dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    plot_results()
