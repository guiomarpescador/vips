import numpy as np
import tensorflow as tf
import gpflow
from src.init_methods.OIPS import  OIPS
from src.init_methods.VIPS import VIPS
from src.init_methods.Offline_VIPS import Offline_VIPS
from utils_magnetometer import save_model

from src.models.osgpr  import OSGPR_VFE
from gpflow.models import SGPR
from src.init_methods.MDPP import ConditionalVariance


def update_parameter(model):
    Zopt = model.inducing_variable.Z.numpy()
    mu_old, Su_old= model.predict_f(Zopt, full_cov=True)
    if len(Su_old.shape) == 3:
        Su_old = Su_old[0, :, :] + 1e-6 * np.eye(mu_old.shape[0])
    Kaa_old = model.kernel(model.inducing_variable.Z)
    sigma2 = model.likelihood.variance.numpy() 
    return Zopt, mu_old, Su_old, Kaa_old, sigma2
    
def run_optimization(model, **opt):
    gpflow.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables)

def evaluation_metrics(model, X_test, y_test):
    """
    Calculate and return RMSE and NLPD
    """
    f_mean, f_var = model.predict_f(X_test)
    if len(f_var.shape) == 1:
        f_var = f_var[..., None]
    y_pred = model.likelihood.predict_mean_and_var(X_test, f_mean, f_var)[0]
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    nlpd = -1 * tf.reduce_mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test)).numpy().item()
    return rmse, nlpd

def initial_step(Xi, yi, method, **method_params):
    params = None
    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(), gpflow.kernels.Matern52()])
    kernel.kernels[0].variance.assign(500)
    if method == 'VIPS':
        delta = method_params["delta"]
        initial_z = Offline_VIPS(Xi, yi, kernel, 0.1).get_inducing_points(delta = delta)[0]
        count = yi.shape[0]
        mean = np.mean(yi)
        M2 = np.sum((yi-mean)**2)
        params = [count, mean, M2]
    elif method == 'OIPS':
        rho = method_params["rho"]
        oips = OIPS(rho_accept=rho)
        initial_z = oips.select_inducing_points(Xi, kernel)
    else:
        M = method_params["M"]
        eta = method_params["eta"]
        initial_z =  ConditionalVariance(threshold=eta, sample=False).compute_initialisation(Xi, M, kernel)[0]
    model =  SGPR((Xi,yi), kernel, initial_z, noise_variance=0.1)
    gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def update_step(method, Xi, yi, model, Z_old, mu_old, Su_old, Kaa_old, sigma2, params = None, **method_params):
    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(model.kernel.kernels[0].variance),
                                 gpflow.kernels.Matern52(
                                     lengthscales=model.kernel.kernels[1].lengthscales,
                                     variance=model.kernel.kernels[1].variance)])
            
    if method == 'VIPS':
        delta = method_params["delta"]
        Z_init = VIPS(Xi, yi, kernel, Z_old, mu_old, Su_old, Kaa_old, sigma2, params)
        Z_new, params = Z_init.update_inducing_points(delta = delta)
    elif method == 'OIPS':
        rho = method_params["rho"]
        Z_init = OIPS(rho_accept=rho)
        Z_new = Z_init.update_inducing_points(Z_old, Xi, kernel)
    else: 
        M = method_params["M"]
        eta = method_params["eta"]
        Xs = np.vstack((Xi, Z_old))
        Z_new = ConditionalVariance(threshold=eta, sample=False).compute_initialisation(Xs, M, kernel)[0]
    model = OSGPR_VFE((Xi,yi), kernel, mu_old, Su_old, Kaa_old, Z_old, Z_new)
    model.likelihood.variance.assign(sigma2)
    gpflow.set_trainable(model.inducing_variable, False)
    return model, params

def run_regression_streaming(data_train, data_test, method, save_batch = False, **method_params):
    no_batches = len(data_train)
    rmse = np.zeros(no_batches)
    nlpd = np.zeros(no_batches)
    params = None

    for i in range(no_batches):
        new_data = data_train[i]
        X, y = (new_data[0], new_data[1])
        if i == 0:
            model, params = initial_step(X, y, method, **method_params)
        else:
            model, params = update_step(method, X, y, model, Zopt, mu_old, Su_old, Kaa_old, sigma2, params=params, **method_params)
        run_optimization(model)
        Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
        rmse[i], nlpd[i] = evaluation_metrics(model, data_test[0], data_test[1])

        if (i != 0) and (i+1) % 5 != 0 or save_batch == False:
            continue
        else:
            save_model(model, method + "_model_path_3_batch_" + str(i))
    return rmse, nlpd, model, params

def run_regression_sequential(data_train, data_test, method, model, params=None, **method_params):
    Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
    
    no_batches = len(data_train)
    rmse = np.zeros(no_batches)
    nlpd = np.zeros(no_batches)

    for i in range(no_batches):
        new_data = data_train[i]
        X, y = (new_data[0], new_data[1])
        model, params = update_step(method, X, y, model, Zopt, mu_old, Su_old, Kaa_old, sigma2, params=params, **method_params)
        run_optimization(model)
        Zopt, mu_old, Su_old, Kaa_old, sigma2 = update_parameter(model)
        rmse[i], nlpd[i] = evaluation_metrics(model, data_test[0], data_test[1])

    return rmse, nlpd, model 

def run_regression_full(data_train, data_test, save_batch = False):
    no_batches = len(data_train)
    params = None
    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(), gpflow.kernels.Matern52()])
    kernel.kernels[0].variance.assign(500)

    ## Get all train data
    X = np.vstack([data[0] for data in data_train])
    y = np.vstack([data[1] for data in data_train])
    model = gpflow.models.GPR(data=(X, y), kernel=kernel, noise_variance=0.1)
    run_optimization(model)
    rmse, nlpd = evaluation_metrics(model, data_test[0], data_test[1])

    save_model(model, "full_model_path_3_batch_" + str(no_batches))
    return rmse, nlpd, model, params