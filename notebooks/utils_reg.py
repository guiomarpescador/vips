import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow

from src.init_methods.VIPS import VIPS
from src.init_methods.Offline_VIPS import Offline_VIPS
from src.models.osgpr  import OSGPR_VFE
from gpflow.models import SGPR, GPR
from gpflow.kernels import SquaredExponential
from matplotlib import ticker
from src.init_methods.MDPP import ConditionalVariance
from src.init_methods.OIPS import OIPS

def plot_model(model, ax, cur_x, cur_y, pred_x, model_full = None, seen_x=None, seen_y=None, plot_mean = True, plot_samples = False, color = 'C0'):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old= model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old= Su_old[0, :, :] + 1e-6 * np.eye(mu_old.shape[0])
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
    ax.set_ylabel("$y$")

def RMSE(model, X_test, y_test):
    """
    Calculate and return root mean squared error (RMSE).
    """
    f_mean, f_var = model.predict_f(X_test)
    if len(f_var.shape) == 1:
        f_var = f_var[..., None]
    y_pred = model.likelihood.predict_mean_and_var(X_test, f_mean, f_var)[0]
    return np.sqrt(np.mean(np.square(y_pred - y_test)))

def NLPD(model, X_test, y_test):
    """
    Calculate and return negative log predictive density (NLPD).
    """
    f_mean, f_var = model.predict_f(X_test)
    if len(f_var.shape) == 1:
        f_var = f_var[..., None]
    return -1 * tf.reduce_mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test)).numpy().item()

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

def init_Z(method, X, y,  kernel, **kwargs):
    params = 0
    if method == "Gradient":
        M = kwargs['M']
        Z = X[np.random.permutation(X.shape[0])[0:M], :]
    elif method == "OIPS":
        Z_init = OIPS(rho_accept=0.80) 
        rng = np.random.default_rng()
        Z = Z_init.select_inducing_points(X, kernel)
    elif method == "CV":
        eta = kwargs['eta']
        M = kwargs['M']
        Z_init = ConditionalVariance(sample = False, threshold = eta)
        Z = Z_init.compute_initialisation(X, M, kernel)[0]
    elif method == "VIPS":
        delta = kwargs['delta']
        Z, params = Offline_VIPS(X, y, kernel, 0.1).get_inducing_points(delta)
    return Z, params

def update_Z(method, X, y, kernel, Z_old, mu_old = None, Su_old = None, Kaa_old = None, sigma2 = None,  params = None, **kwargs):
    if method == 'CV':
        eta = kwargs['eta']
        M = kwargs['M']
        Z_init = ConditionalVariance(sample = False, threshold = eta)
        Xs = np.vstack((Z_old, X))
        Z = Z_init.compute_initialisation(Xs, M, kernel)[0]
    elif method == 'OIPS':
        Z_init = OIPS(rho_accept=0.80)
        Z = Z_init.update_inducing_points(Z_old, X, kernel)
    elif method == 'Gradient':
        M = Z_old.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = Z_old[np.random.permutation(Z_old.shape[0])[0:M_old], :]
        new_Z = X[np.random.permutation(X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    elif method == 'VIPS':
        delta = kwargs['delta']
        Z_init = VIPS(X, y, kernel, Z_old, mu_old, Su_old, Kaa_old, sigma2, params)
        Z, params = Z_init.update_inducing_points(delta = delta)
    return Z, params

def update_parameter(model):
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old= model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old= Su_old[0, :, :] + 1e-8 * np.eye(mu_old.shape[0])
    Kaa_old = model.kernel(model.inducing_variable.Z)
    sigma2 = model.likelihood.variance.numpy() 
    return Zopt, mu_old, Su_old, Kaa_old, sigma2

def run_optimization(model):
    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables)

def init_step(Xi, yi, kernel, method, **kwargs):
    Z_init, params = init_Z(method, Xi, yi, kernel, **kwargs)
    model =  SGPR((Xi,yi), kernel, Z_init, noise_variance=0.1)
    if method != 'Gradient':
        gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def update_step(Xi, yi, kernel, method, mu_old, Su_old, Kaa_old, Z_old, sigma2, params, **kwargs):
    Z_new, params = update_Z(method, Xi, yi, kernel, Z_old, mu_old, Su_old, Kaa_old, sigma2, params, **kwargs)
    model = OSGPR_VFE((Xi, yi), kernel, mu_old, Su_old, Kaa_old, Z_old, Z_new)
    model.likelihood.variance.assign(sigma2)
    if method != 'Gradient':
        gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def run_regression(X, y, xx, no_batches, method, **kwargs):

    model_full = GPR((X, y), SquaredExponential(variance=1.0, lengthscales=0.5))
    run_optimization(model_full)

    fig, axs = plt.subplots(no_batches, 1, figsize=(20,10), dpi=200)
    axs[0].set_title(f"Regression example (N={len(X)})", fontsize=8)

    X_train, y_train = div_data(X, y, no_batches)

    for i in range(no_batches):
        Xi = X_train[i]
        yi = y_train[i]
        if i == 0:
            kernel = SquaredExponential(variance=1.0, lengthscales=0.5)
            model, params = init_step(Xi, yi, kernel, method, **kwargs)
            seen_x = None
            seen_y = None
        else:
            model, params = update_step(Xi, yi, model.kernel, method, mu_old, Su_old, Kaa_old, Zopt, sigma2, params, **kwargs)
            seen_x = np.vstack(X_train[:i])
            seen_y = np.vstack(y_train[:i])
        
        run_optimization(model)
        print(f"Batch {i+1} completed")
        Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
        plot_model(model, axs[i], Xi, yi, xx, model_full, seen_x, seen_y, plot_samples = False)
        
    axs[-1].set_xlabel("Regression input $x$", fontsize=8)
    axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=8)
    fig.tight_layout()
    plt.savefig(f"figures/{method}_regression.pdf")
    return model
    

def run_exact(X, y, X_test, y_test, no_batches, **options):
    X_train, y_train = div_data(X, y, no_batches)
    rmse = np.zeros(no_batches)
    nlpd = np.zeros(no_batches)
    for i in range(no_batches):
        Xi = np.vstack(X_train[:i+1])
        yi = np.vstack(y_train[:i+1])
        model = GPR((Xi, yi), SquaredExponential(variance=1.0, lengthscales=0.5))
        run_optimization(model, **options)
        rmse[i] = RMSE(model, X_test, y_test)
        nlpd[i] = NLPD(model, X_test, y_test)
    results = np.round(rmse,2), np.round(nlpd,2), np.zeros(no_batches)
    return results
