import numpy as np
import gpflow

from src.init_methods.VIPS import VIPS
from src.init_methods.Offline_VIPS import Offline_VIPS
from src.models.osgpr  import OSGPR_VFE
from gpflow.models import SGPR
from gpflow.kernels import SquaredExponential
from src.init_methods.MDPP import ConditionalVariance


def init_Z(method, X, y,  kernel, **kwargs):
    params = 0
    if method == "Gradient":
        M = kwargs['M']
        Z = X[np.random.permutation(X.shape[0])[0:M], :]
    elif method == "CV":
        eta = kwargs['eta']
        M = kwargs['M']
        Z_init = ConditionalVariance(sample = False, threshold = eta)
        Z = Z_init.compute_initialisation(X, M, kernel)[0]
    elif "VIPS" in method:
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
    elif method == 'Gradient':
        M = Z_old.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = Z_old[np.random.permutation(Z_old.shape[0])[0:M_old], :]
        new_Z = X[np.random.permutation(X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    elif "VIPS" in method:
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

def run_regression(X_train, y_train, no_batches, method, **kwargs):
    Ms = np.zeros(no_batches)
    for i in range(no_batches):
        Xi = X_train[i]
        yi = y_train[i]
        if i == 0:
            kernel = SquaredExponential(variance=1.0, lengthscales=0.5)
            model, params = init_step(Xi, yi, kernel, method, **kwargs)
        else:
            model, params = update_step(Xi, yi, model.kernel, method, mu_old, Su_old, Kaa_old, Zopt, sigma2, params, **kwargs)
        
        run_optimization(model)
        Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
        Ms[i] = int(Zopt.shape[0])
    return model, Ms

