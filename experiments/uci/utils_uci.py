import time
from typing import Sequence, Union
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp

from src.init_methods.Offline_VIPS import Offline_VIPS
from src.models.osgpr  import OSGPR_VFE
from gpflow.models import SGPR, GPR
from gpflow.kernels import SquaredExponential
from src.init_methods.VIPS import VIPS
from src.init_methods.MDPP import ConditionalVariance
from src.init_methods.OIPS import OIPS
from utils_data import *


datasets = {
    "bike": Bike(),
    "concrete": Concrete(),
    "elevators": Elevators(),
    "kin8nm": Kin8nm(),
    "skillcraft": Skillcraft(), 
    "naval": Naval(), 
    "naval_time": Naval(),
}

def load_data(dataset_name: str, train_split_percentage: float = 0.8, n_k_folds: int = None, seed: int = 0,  no_batches: int = 10, order: bool = True):
    """
    Load the dataset and split it into train and test sets.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = datasets[dataset_name]
    X, y = dataset.X, dataset.y

    if n_k_folds is None:
        X_train_batches, y_train_batches, X_test_batches, y_test_batches = divide_batches(X, y, no_batches, train_split_percentage, order)
        train_data = [(X_train_batches, y_train_batches)]
        test_data = [(X_test_batches, y_test_batches)]
    else:
        X_train_batches, y_train_batches, X_test_batches, y_test_batches = divide_batches_kfolds(X, y, no_batches, n_k_folds, order)
        train_data = [(X_train_batches[i], y_train_batches[i]) for i in range(n_k_folds)]
        test_data = [(X_test_batches[i], y_test_batches[i]) for i in range(n_k_folds)]

    return train_data, test_data

def divide_batches_kfolds(X, y, no_batches, n_k_folds=5, order=True):
    if order:
        idx = np.argsort(X[:, 0])  # Sort by the first column
        X = X[idx, :]
        y = y[idx, :]
    else:
        idx = np.random.permutation(X.shape[0])
        X = X[idx, :]
        y = y[idx, :]

    X_train_batches_folds = [[] for _ in range(n_k_folds)]
    y_train_batches_folds = [[] for _ in range(n_k_folds)]
    X_test_batches_folds = [[] for _ in range(n_k_folds)]
    y_test_batches_folds = [[] for _ in range(n_k_folds)]

    N = X.shape[0]
    batch_size = int(N // no_batches)

    kf = KFold(n_splits=n_k_folds, shuffle=False)

    # Loop over each batch
    for i in range(no_batches):
        # Split the data into batches
        if i == no_batches - 1:
            X_batch = X[(no_batches - 1) * batch_size:, :]
            y_batch = y[(no_batches - 1) * batch_size:, :]
        else:
            X_batch = X[i * batch_size: (i + 1) * batch_size, :]
            y_batch = y[i * batch_size: (i + 1) * batch_size, :]

        # Shuffle the current batch
        indices = np.random.permutation(X_batch.shape[0])
        X_batch = X_batch[indices]
        y_batch = y_batch[indices]

        # Now apply k-fold within this batch
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_batch)):
            X_train_batches_folds[fold_idx].append(X_batch[train_idx])
            y_train_batches_folds[fold_idx].append(y_batch[train_idx])
            X_test_batches_folds[fold_idx].append(X_batch[test_idx])
            y_test_batches_folds[fold_idx].append(y_batch[test_idx])

    return X_train_batches_folds, y_train_batches_folds, X_test_batches_folds, y_test_batches_folds

def divide_batches(X, y, no_batches, train_fraction=0.8, order=True):
    
    if order:
        idx = np.argsort(X[:, 0])
        X = X[idx, :]
        y = y[idx, :]
    else:
        idx = np.random.permutation(X.shape[0])
        X = X[idx, :]
        y = y[idx, :]

    X_train_batches = []
    y_train_batches = []
    X_test_batches = []
    y_test_batches = []
    
    N = X.shape[0]
    batch_size = int(N / no_batches)
    # Split into train and test sets
    train_size = int((X.shape[0]//no_batches)*train_fraction)

    
    for i in range(no_batches - 1):
        X_batch = X[i * batch_size: (i + 1) * batch_size, :]
        y_batch = y[i * batch_size: (i + 1) * batch_size, :]
        # Shuffle the current batch
        indices = np.random.permutation(X_batch.shape[0])
        X_batch = X_batch[indices]
        y_batch = y_batch[indices]

        X_train_batches.append(X_batch[:train_size])
        y_train_batches.append(y_batch[:train_size])
        X_test_batches.append(X_batch[train_size:])
        y_test_batches.append(y_batch[train_size:])

    # Last batch
    X_batch = X[(no_batches - 1) * batch_size:, :]
    y_batch = y[(no_batches - 1) * batch_size:, :]
    indices = np.random.permutation(X_batch.shape[0])
    X_batch = X_batch[indices]
    y_batch = y_batch[indices]
    X_train_batches.append(X_batch[:train_size])
    y_train_batches.append(y_batch[:train_size])
    X_test_batches.append(X_batch[train_size:])
    y_test_batches.append(y_batch[train_size:])

    return X_train_batches, y_train_batches, X_test_batches, y_test_batches

class RegressionMethod:
    def __init__(self, X, y, no_batches, X_test, y_test, test_per_batch = False, **method_params):
        self.X = X
        self.y = y
        self.no_batches = no_batches
        self.X_test = X_test
        self.y_test = y_test
        self.params = method_params
        self.test_per_batch = test_per_batch
        
    def evaluate(self, model, batch):
        if self.test_per_batch == True:
            X_test, y_test = np.vstack(self.X_test[:batch]), np.vstack(self.y_test[:batch])
        else:
            X_test, y_test = np.vstack(self.X_test), np.vstack(self.y_test)
        f_mean, f_var = model.predict_f(X_test)
        if len(f_var.shape) == 1:
            f_var = f_var[..., None]
        rmse = np.sqrt(np.mean((f_mean - y_test) ** 2))
        nlpd = -1 * tf.reduce_mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test)).numpy().item()    
        return np.round(rmse, 4), np.round(nlpd, 4)
    
    def hyperparameter_initialisation(self):
        lengthscale = 1.0
        variance = 1.0
        noise = 0.1
        return [variance, lengthscale, noise]
    
    def update_parameter(self, model):
        Zopt = model.inducing_variable.Z.numpy()
        ma_old, Saa_old = model.predict_f(Zopt, full_cov=True)
        if len(Saa_old.shape) == 3:
            Saa_old = Saa_old[0, :, :] + 1e-6 * tf.eye(Saa_old.shape[1], dtype=tf.float64)
        Kaa_old = model.kernel(model.inducing_variable.Z)
        sigma2 = model.likelihood.variance.numpy() 
        return Zopt, ma_old, Saa_old, Kaa_old, sigma2

    def create_results_dict(self, time = None, rmse = None, nlpd = None, n_batch = None, m_batch = None):
        results =   {
            'time': time,
            'RMSE': rmse,
            'NLPD': nlpd,
            'N': n_batch,
            'M': m_batch
        }
        # Remove any empty lists
        results = {k: v for k, v in results.items() if v is not None}
        return results
    
    def run_optimization(self, model):
        bfgs_optimizer = gpflow.optimizers.Scipy()
        bfgs_optimizer.minimize(model.training_loss, model.trainable_variables)


    def run(self):
        raise NotImplementedError("Subclasses should implement this!")

class run_CV(RegressionMethod):
    def run(self):
        np.random.seed(0)
        X_batch, y_batch = self.X, self.y
        no_batches = self.no_batches
        eta, M = self.params['eta'], self.params['M']
        time_condvar, rmse_condvar, nlpd_condvar, n_batch, m_batch = [], [], [], [], []

        
        for i in range(no_batches):
            start_time = time.time()
            Xi = X_batch[i]
            yi = y_batch[i]
            print('Batch: ', i)
            if i == 0:
                hyperparameters = self.hyperparameter_initialisation()
                kernel = SquaredExponential(variance=hyperparameters[0], lengthscales=hyperparameters[1])
                cond_var = ConditionalVariance(threshold=eta, sample=False)
                Z_init = cond_var.compute_initialisation(Xi, M, kernel)[0]
                model_condvar = SGPR((Xi, yi), kernel, Z_init, noise_variance=hyperparameters[2])
            else:
                kernel = SquaredExponential(variance=model_condvar.kernel.variance, lengthscales=model_condvar.kernel.lengthscales)
                Xs = np.vstack((Z_opt, Xi))
                Z_new = cond_var.compute_initialisation(Xs, M, kernel)[0]
                model_condvar = OSGPR_VFE((Xi, yi), kernel, ma_old, Saa_old, Kaa_old, Z_opt, Z_new)
                model_condvar.likelihood.variance.assign(sigma2)
            model_condvar.kernel.variance = gpflow.Parameter(model_condvar.kernel.variance, transform=tfp.bijectors.Sigmoid(low=tf.cast(1e-6, tf.float64), high=tf.cast(2e3, tf.float64)))
            gpflow.set_trainable(model_condvar.inducing_variable, False)
            self.run_optimization(model_condvar)
            Z_opt, ma_old, Saa_old, Kaa_old, sigma2 = self.update_parameter(model_condvar)
            end_time = time.time()
            time_condvar.append(end_time - start_time)
            rmse, nlpd = self.evaluate(model_condvar, i+1)
            rmse_condvar.append(rmse)
            nlpd_condvar.append(nlpd)
            n_batch.append(Xi.shape[0])
            m_batch.append(Z_opt.shape[0])   
    
        results = self.create_results_dict(time_condvar, rmse_condvar, nlpd_condvar, n_batch, m_batch)
    
        return results

class run_OIPS(RegressionMethod):
    def run(self):
        np.random.seed(0)
        X_batch, y_batch = self.X, self.y
        no_batches = self.no_batches
        rho = self.params['rho']
        oips = OIPS(rho_accept=rho)
        time_oips, rmse_oips, nlpd_oips, n_batch, m_batch = [], [], [], [], []

        for i in range(no_batches):
            start_time = time.time()
            Xi = X_batch[i]
            yi = y_batch[i]
            print('Batch: ', i)
            if i == 0:
                hyperparameters = self.hyperparameter_initialisation()
                kernel = SquaredExponential(variance=hyperparameters[0], lengthscales=hyperparameters[1])
                Z_init = oips.select_inducing_points(Xi, kernel)  
                model_oips  = SGPR((Xi,yi), kernel, Z_init, noise_variance=hyperparameters[2])
            else:
                kernel = SquaredExponential(variance=model_oips.kernel.variance, lengthscales=model_oips.kernel.lengthscales)
                Z_new = oips.update_inducing_points(Z_opt, Xi, kernel)    
                model_oips = OSGPR_VFE((Xi, yi), kernel, ma_old, Saa_old, Kaa_old, Z_opt, Z_new)
                model_oips.likelihood.variance.assign(sigma2)
            gpflow.set_trainable(model_oips.inducing_variable, False)
            self.run_optimization(model_oips)
            Z_opt, ma_old, Saa_old, Kaa_old, sigma2 = self.update_parameter(model_oips)
            end_time = time.time()
            time_oips.append(end_time - start_time)
            rmse, nlpd = self.evaluate(model_oips, i+1)
            rmse_oips.append(rmse)
            nlpd_oips.append(nlpd)
            n_batch.append(Xi.shape[0])
            m_batch.append(Z_opt.shape[0])
        results = self.create_results_dict(time_oips, rmse_oips, nlpd_oips, n_batch, m_batch)
        return results
    

class run_VIPS(RegressionMethod):
    def run(self):
        np.random.seed(0)
        X_batch, y_batch = self.X, self.y
        no_batches = self.no_batches
        delta = self.params['delta']
        time_vips, rmse_vips, nlpd_vips, n_batch, m_batch = [], [], [], [], []

        for i in range(no_batches):
            start_time = time.time()
            Xi = X_batch[i]
            yi = y_batch[i]
            print('Batch: ', i+1)
            if i == 0:
                hyperparameters = self.hyperparameter_initialisation()
                kernel = SquaredExponential(variance=hyperparameters[0], lengthscales=hyperparameters[1])
                sigma2 = hyperparameters[2]
                vips = Offline_VIPS(Xi, yi, kernel, sigma2)
                Z, params = vips.get_inducing_points(delta=delta)
                model_vips =  SGPR((Xi,yi), kernel, Z, noise_variance=sigma2)
            else:
                kernel = SquaredExponential(variance=model_vips.kernel.variance, lengthscales=model_vips.kernel.lengthscales)
                vips = VIPS(Xi, yi, kernel, Z_opt, ma_old, Saa_old, Kaa_old, sigma2, params)
                Z_new, params = vips.update_inducing_points(delta = delta)
                model_vips = OSGPR_VFE((Xi, yi), kernel, ma_old, Saa_old, Kaa_old, Z_opt, Z_new)
                model_vips.likelihood.variance.assign(sigma2)
            model_vips.kernel.variance = gpflow.Parameter(model_vips.kernel.variance, transform=tfp.bijectors.Sigmoid(low=tf.cast(1e-6, tf.float64), high=tf.cast(2e3, tf.float64)))
            gpflow.set_trainable(model_vips.inducing_variable, False)
            self.run_optimization(model_vips)
            Z_opt, ma_old, Saa_old, Kaa_old, sigma2 = self.update_parameter(model_vips)   
            end_time = time.time()
            time_vips.append(end_time - start_time)
            rmse, nlpd = self.evaluate(model_vips, i+1)
            rmse_vips.append(rmse)
            nlpd_vips.append(nlpd)
            n_batch.append(Xi.shape[0])
            m_batch.append(Z_opt.shape[0])
        results = self.create_results_dict(time_vips, rmse_vips, nlpd_vips, n_batch, m_batch)

        return results
    
class Full_GP(RegressionMethod):
    def run(self):
        np.random.seed(0)
        X_batch, y_batch = self.X, self.y
        no_batches = self.no_batches
        rmse_full, nlpd_full, n_batch_full = [], [], []   

        index = [0, int(no_batches/2)-1, no_batches]
        hyper_init = self.hyperparameter_initialisation()
        kernel_gpr = SquaredExponential(variance=hyper_init[0], lengthscales=hyper_init[1])
        sigma2 = hyper_init[2]

        for i in index:
            Xi = np.vstack(X_batch[:i+1])
            yi = np.vstack(y_batch[:i+1])
            model_gpr = GPR((Xi, yi), kernel_gpr, noise_variance=sigma2)
            self.run_optimization(model_gpr)
            rmse, nlpd = self.evaluate(model_gpr, i+1)
            rmse_full.append(rmse)
            nlpd_full.append(nlpd)
            n_batch_full.append(Xi.shape[0])
    
        results = self.create_results_dict(rmse = rmse_full, nlpd = nlpd_full, n_batch = n_batch_full, m_batch = None)
        return results
    