import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import utils_reg as utils
from gpflow.kernels import SquaredExponential
from gpflow.models import GPR

# Set the font to times new roman
plt.rcParams["font.family"] = "Times New Roman"


def plot_model(model, ax, cur_x, cur_y, pred_x, model_full = None, seen_x=None, seen_y=None, plot_mean = True, plot_samples = False, color = 'C0'):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old = model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old= Su_old[0, :, :]
        vx = vx[:, 0]

    ax.plot(cur_x, cur_y, 'x', color = 'C1', mew=1, alpha=0.8, label='Data')
    
    if seen_x is not None:
        ax.plot(seen_x, seen_y, 'kx', mew=1, alpha=0.2, label='Seen data')
    
    if plot_mean:
        ax.plot(pred_x, mx, lw=1.5, label="Mean", color = color)

    if plot_samples:
        f_samples = model.predict_f_samples(pred_x, 10)
        ax.plot(pred_x,f_samples[:, :, 0].numpy().T, color='C0', alpha=0.4, lw=0.5)

    ax.fill_between(
        pred_x[:, 0], mx[:, 0] -  2*np.sqrt(vx), 
        mx[:, 0] +  2*np.sqrt(vx),  alpha=0.3, label="Approx. Post.", color = color)
    
    ax.plot(Zopt, mu_old, 'o', color='k', mew=2, label="Inducing Points")

    if model_full is not None: 
        mx_full, vx_full = model_full.predict_f(pred_x)
        ax.plot(pred_x, mx_full + 2.0 * vx_full ** 0.5, color='k', linestyle='--', linewidth=2.0)
        ax.plot(pred_x, mx_full - 2.0 * vx_full** 0.5, color='k', linestyle='--', linewidth=2.0, label="Full GP")

    ax.set_ylim([-2, 2 ])
    ax.set_xlim([np.min(pred_x), np.max(pred_x)])
    plt.subplots_adjust(hspace = .08)
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)


def func(x):
    # random fourier features
    n = x.shape[0]
    return np.sin(2*x) + np.cos(5*x) + np.random.normal(0, 0.2, n)

def div_data(X, y, no_batches):

    N = X.shape[0]
    mb_size = int(np.floor(N/no_batches))

    X_batch = []
    y_batch = []

    for i in range(no_batches-1):
        X_batch.append(X[i*mb_size:(i+1)*mb_size, :])
        y_batch.append(y[i*mb_size:(i+1)*mb_size, :])

    X_batch.append(X[(i+1)*mb_size:, :])
    y_batch.append(y[(i+1)*mb_size:, :])

    return X_batch, y_batch


def run_experiment():
    # X randm and ordered 
    np.random.seed(0)
    method = 'VIPS'
    parameters = {'delta': 0.05}

    # Dataset 1 
    no_batches = 10
    n = 500
    X = np.random.uniform(0, 10, n)
    X = np.sort(X)
    y = func(X)

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X_batch_1, y_batch_1 = div_data(X, y, no_batches)
    _, M_dataset_1 = utils.run_regression(X_batch_1, y_batch_1, no_batches, method, **parameters)

    full_model_1 = GPR((X, y), kernel=SquaredExponential(), noise_variance=0.5)
    utils.run_optimization(full_model_1)

    # Dataset 2
    n = 150
    no_batches = 10
    X = np.random.uniform(0, 10, n)
    y = func(X)

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X_batch_2, y_batch_2 = div_data(X, y, no_batches)

    _, M_dataset_2 = utils.run_regression(X_batch_2, y_batch_2, no_batches, method, **parameters)

    full_model_2 = GPR((X, y), kernel=SquaredExponential(), noise_variance=0.5)
    utils.run_optimization(full_model_2)
    
    # Dataset 3
    # Data concentrated in the middle
    n = 1000
    no_batches = 10
    X_uni = np.random.uniform(4.5, 7, n)
    X_cauchy = np.random.standard_cauchy(300) + 6
    X_cauchy = X_cauchy[(X_cauchy > 0) & (X_cauchy < 10)]


    X = np.concatenate((X_uni, X_cauchy))
    y = func(X)
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X_batch_3, y_batch_3 = div_data(X, y, no_batches)
    _, M_dataset_3 = utils.run_regression(X_batch_3, y_batch_3, no_batches, method, **parameters)

    full_model_3 = GPR((X, y), kernel=SquaredExponential(), noise_variance=0.5)
    utils.run_optimization(full_model_3)

    # Plot each batch with different colors
    fig, axs = plt.subplots(1, 3, figsize=(14, 3), dpi=300, sharex=True, sharey=True)

    colors = ['#00008b',  '#6495ed', '#d95b43']
   # colors = ['#7030a0', '#53777a','#d95b43']
    xx = np.linspace(0, 12, 1000).reshape(-1, 1)

    mx, vx = full_model_1.predict_f(xx)
    axs[0].plot(xx, mx, color = 'lightgrey', label='GP mean')
    for i in range(no_batches):
        Xi, yi = X_batch_1[i], y_batch_1[i]
        axs[0].plot(Xi, yi, 'x', alpha=0.8, label='Data points', color = colors[0])
    
    axs[0].set_title("Dataset 1", fontsize=14)
    axs[0].set_ylabel("y", fontsize=14)
    axs[0].set_xlabel("x", fontsize=14)

    mx, vx = full_model_2.predict_f(xx)
    axs[1].plot(xx, mx, color='lightgrey')
    for i in range(no_batches):
        Xi, yi = X_batch_2[i], y_batch_2[i]
        axs[1].plot(Xi, yi, 'x', mew=1, alpha=0.8, color =  colors[0])
    axs[1].set_title("Dataset 2", fontsize=14)
    axs[1].set_xlabel("x", fontsize=14)

    mx, vx = full_model_3.predict_f(xx)
    axs[2].plot(xx, mx, color='lightgrey')
    for i in range(no_batches):
        Xi, yi = X_batch_3[i], y_batch_3[i]
        axs[2].plot(Xi, yi, 'x', mew=1, alpha=0.8, color = colors[0])
    axs[2].set_title("Dataset 3", fontsize=14)
    axs[2].set_xlabel("x", fontsize=14)
    axs[2].limits = [0, 10]

    # get legend elements from first plot 
    handles, _ = axs[0].get_legend_handles_labels()

    labels = ['Exact GP Mean', 'Data Points']

    # Create a legend for the first plot
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=10, fontsize=15)
    
    # handles, labels = axs[3].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=14)

    plt.tight_layout()

    plt.savefig('experiments/data_types/syntetic_batches.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(2, 3, figsize=(12, 5), sharey='row')
    # set label size for all plots
    for ax in axs.flatten():
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)

    # Plot the first dataset
    mx, vx = full_model_1.predict_f(xx)
    axs[0, 0].plot(xx, mx, color = 'lightgrey', label='GP mean')
    for i in range(3):
        Xi, yi = X_batch_1[i], y_batch_1[i]
        axs[0,0].plot(Xi, yi, 'x', alpha=0.8, label='Batch {}'.format(i+1), color = colors[i])
    
    axs[0, 0].set_title("Dataset 1", fontsize=18)
    axs[0, 0].set_ylabel("y", fontsize=18)
    axs[0, 0].set_xlabel("x", fontsize=18)
    # Limit the x axis
    axs[0, 0].set_xlim([0, np.max(X_batch_1[2])])

    # Plot the second dataset
    mx, vx = full_model_2.predict_f(xx)
    axs[0, 1].plot(xx, mx, color='lightgrey')
    for i in range(3):
        Xi, yi = X_batch_2[i], y_batch_2[i]
        axs[0, 1].plot(Xi, yi, 'x', mew=1, alpha=0.8, label='Batch {}'.format(i+1), color = colors[i])
    axs[0, 1].set_title("Dataset 2", fontsize=18)
    axs[0, 1].set_xlabel("x", fontsize=18)
    # Make number on the x-axis bigger

    # Plot the third dataset
    mx, vx = full_model_3.predict_f(xx)
    axs[0, 2].plot(xx, mx, color='lightgrey')
    for i, batch in enumerate([1, 2, -1]):
        Xi, yi = X_batch_3[batch], y_batch_3[batch]
        axs[0, 2].plot(Xi, yi, 'x', mew=1, alpha=0.8, color = colors[i])

    axs[0, 2].set_title("Dataset 3", fontsize=18)
    axs[0, 2].set_xlabel("x", fontsize=18)
    axs[0, 2].limits = [0, 10]

    # Plot the indcuing points for the first dataset
    axs[1, 0].plot(np.arange(1, M_dataset_1.shape[0]+1), M_dataset_1, '+-', label = 'Inducing Points', color = '#800080')
    axs[1, 0].set_xlabel("Batch", fontsize=18)
    axs[1, 0].set_ylabel("Inducing Points", fontsize=18)


    # Plot the indcuing points for the second dataset
    axs[1, 1].plot(np.arange(1, M_dataset_2.shape[0]+1), M_dataset_2, '+-', color = '#800080')
    axs[1, 1].set_xlabel("Batch", fontsize=18)

    # Plot the inducing points
    axs[1, 2].plot(np.arange(1, M_dataset_3.shape[0]+1), M_dataset_3, '+-', color = '#800080')
    axs[1, 2].set_xlabel("Batch", fontsize=18)


    handles, _ = axs[0, 0].get_legend_handles_labels()
    handles_M, _ = axs[1, 0].get_legend_handles_labels()
    handles.extend(handles_M)

    labels = ['Exact GP Mean', 'Past batch', 'Current batch', 'Future batch', 'VIPS']
         
    # Put legend on below the plot
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=15)

    plt.tight_layout()
    plt.savefig('experiments/data_types/syntetic_batches_2.pdf', bbox_inches='tight')

if __name__ == "__main__":
    run_experiment()