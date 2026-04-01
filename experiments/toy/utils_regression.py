import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow

from src.init_methods.Offline_VIPS import Offline_VIPS
from src.models.osgpr  import OSGPR_VFE
from gpflow.models import SGPR, GPR
from gpflow.kernels import SquaredExponential
from matplotlib import ticker
from src.init_methods.VIPS import VIPS
from src.init_methods.MDPP import ConditionalVariance

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

    # Sort the data
    idx = np.argsort(X[:, 0])
    X = X[idx, :]
    y = y[idx, :]
    
    X_batch = []
    y_batch = []

    for i in range(no_batches-1):
        X_batch.append(X[i*mb_size:(i+1)*mb_size, :])
        y_batch.append(y[i*mb_size:(i+1)*mb_size, :])

    X_batch.append(X[(i+1)*mb_size:, :])
    y_batch.append(y[(i+1)*mb_size:, :])

    return X_batch, y_batch

def plot_model(model, ax, cur_x, cur_y, pred_x, method):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old = model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old= Su_old[0, :, :]+ 1e-6 * np.eye(mu_old.shape[0])
        vx = vx[:, 0]

    methods_dict = {'Gradient': 'Gradient', 'OIPS': 'OIPS', 'CVF_BG': 'CVF-BG', 'MDPP': 'CV'}
    ax.text(0.5, 0.1, "{}, M = {}".format(methods_dict[method], Zopt.shape[0]), transform=ax.transAxes, fontsize=15, color='black')
    X, y = div_data(cur_x, cur_y, 3)
    colors = ['C1', 'C3', 'C6']
    for i in range(3):
        ax.plot(X[i], y[i], 'x', mew=1, alpha=0.8, label='Data batch {}'.format(i+1), color = colors[i])


    ax.plot(pred_x, mx, lw=1.5, label="Mean", color = 'C0')
    
    ax.fill_between(
        pred_x[:, 0], mx[:, 0] -  2*np.sqrt(vx), 
        mx[:, 0] +  2*np.sqrt(vx),  alpha=0.3, label="Approx. Post.", color = 'C0')
    
    ax.plot(Zopt, mu_old, 'o', color='k', mew=2, label="Inducing Points")

    ax.set_ylim([-3, 3])
    ax.set_xlim([np.min(pred_x), np.max(pred_x)])
    plt.subplots_adjust(hspace = .08)
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.set_ylabel("$y$")

def init_Z(method, X, y,  kernel, eta = None, M = None):
    params = 0
    if method == "Gradient":
        Z = X[np.random.permutation(X.shape[0])[0:M], :]
    elif method == "MDPP":
        Z_init = ConditionalVariance(sample = False, threshold = 1e-18)
        Z = Z_init.compute_initialisation(X, M, kernel)[0]
    elif method == 'VIPS':
        vip = Offline_VIPS(X, y, kernel, sigma2=0.5)
        Z, params = vip.get_inducing_points(delta = 0.04)
    return Z, params

def update_Z(method, X, y, kernel, Z_old, eta = None, M = None, mu_old = None, Su_old = None, Kaa_old = None, sigma2 = None, params=None):
    if method == 'MDPP':
        Z_init = ConditionalVariance(sample = False, threshold = eta)
        Xs = np.vstack((Z_old, X))
        Z = Z_init.compute_initialisation(Xs, M, kernel)[0]
    elif method == 'Gradient':
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = Z_old[np.random.permutation(Z_old.shape[0])[0:M_old], :]
        new_Z = X[np.random.permutation(X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    elif method == 'VIPS':
        Z_init = VIPS(X, y, kernel, Z_old, mu_old, Su_old, Kaa_old, sigma2, params)
        Z, params = Z_init.update_inducing_points(delta = 0.05)
    return Z, params

def update_parameter(model):
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old= model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old= Su_old[0, :, :] + 1e-6 * np.eye(mu_old.shape[0])
    Kaa_old = model.kernel(model.inducing_variable.Z)
    sigma2 = model.likelihood.variance.numpy() 
    return Zopt, mu_old, Su_old, Kaa_old, sigma2

def run_optimization(model, **opt):
    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables, options=opt)

def init_step(Xi, yi, kernel, method, eta = None, M = None, **options):
    Z_init, params = init_Z(method, Xi, yi, kernel, eta, M)
    model =  SGPR((Xi,yi), kernel, Z_init, noise_variance=0.5)
    if method != 'Gradient':
        gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def update_step(Xi, yi, kernel, method, mu_old, Su_old, Kaa_old, Z_old, sigma2, eta = None, M = None, params = None, **options):
    Z_new, params = update_Z(method, Xi, yi, kernel, Z_old, eta, M,  mu_old, Su_old, Kaa_old, sigma2, params)
    model = OSGPR_VFE((Xi, yi), kernel, mu_old, Su_old, Kaa_old, Z_old, Z_new)
    model.likelihood.variance.assign(sigma2)
    if method != 'Gradient':
        gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def run_regression(X, y, no_batches, method, test_data = None, eta = None, M = None, **options):
    np.random.seed(42)
    X_train, y_train = div_data(X, y, no_batches)
    rmse = np.zeros(no_batches)
    nlpd = np.zeros(no_batches)
    Ms =  np.zeros(no_batches)
    kernel = SquaredExponential(lengthscales=0.5, variance=1.5)

    for i in range(no_batches):
        Xi = X_train[i]
        yi = y_train[i]
        if i == 0:
            model, params = init_step(Xi, yi, kernel, method, eta, M, **options)
        else:
            model, params = update_step(Xi, yi, model.kernel, method, mu_old, Su_old, Kaa_old, Zopt, sigma2, eta, M, params, **options)
        run_optimization(model, **options)
        Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
        rmse[i] = RMSE(model, test_data[0], test_data[1])
        nlpd[i] = NLPD(model, test_data[0], test_data[1])
        Ms[i] = int(Zopt.shape[0])

    results = np.round(rmse,1), np.round(nlpd,1), Ms # For plotting
    return results

def run_exact(X, y, X_test, y_test, no_batches, **options):
    np.random.seed(42)
    X_train, y_train = div_data(X, y, no_batches)
    rmse = np.zeros(no_batches)
    nlpd = np.zeros(no_batches)
    for i in range(no_batches):
        Xi = np.vstack(X_train[:i+1])
        yi = np.vstack(y_train[:i+1])
        model = GPR((Xi, yi), SquaredExponential(lengthscales=0.5, variance=1.5), noise_variance=0.5)
        run_optimization(model, **options)
        rmse[i] = RMSE(model, X_test, y_test)
        nlpd[i] = NLPD(model, X_test, y_test)
    print(model.kernel.lengthscales)
    print(model.kernel.variance)
    print(model.likelihood.variance)
    results = np.round(rmse,1), np.round(nlpd,1), np.zeros(no_batches) # For plotting
    return results
