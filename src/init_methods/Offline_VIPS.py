import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import add_noise_cov

""" 
Offline Implementation of our algorithm Vegas Inducing Point Selection (VIPS)
"""

class Offline_VIPS:
    def __init__(self, X, y, kernel, sigma2):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.sigma2 = sigma2
        self.common = self.calculate_common_terms()

    def calculate_common_terms(self):

        X, y = self.X, self.y
        N = X.shape[0]
        sigma2 = self.sigma2

        Kff_diag = self.kernel(self.X, full_cov=False)

        # Constant term SGPR bound
        bound = -0.5 * N * np.log(2 * np.pi * sigma2)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(y)) / sigma2

        return Kff_diag, bound

    def calculate_elbo(self, Z):

        X, y = self.X, self.y
        sigma2 = self.sigma2
        sigma = tf.sqrt(sigma2)
        N = X.shape[0]

        Kff_diag, bound = self.common

        M = Z.shape[0]

        jitter = gpflow.utilities.to_default_float(1e-4)

        Kuu = self.kernel(Z) + jitter * tf.eye(M, dtype=gpflow.default_float()) 
        Kuf = self.kernel(Z, X)

        Lu = tf.linalg.cholesky(Kuu)
        Luinv_Kuf = tf.linalg.triangular_solve(Lu, Kuf, lower=True)
        d = tf.linalg.matmul(Luinv_Kuf, Luinv_Kuf, transpose_b=True)

        D = tf.eye(M, dtype=gpflow.default_float()) + d/sigma2
        D = add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        c = tf.matmul(Kuf, y) / sigma2
        Luinv_c = tf.linalg.triangular_solve(Lu, c, lower=True)
        LDinv_Luinv_c = tf.linalg.triangular_solve(LD, Luinv_c, lower=True)
        Qff = tf.linalg.matmul(Luinv_Kuf, Luinv_Kuf, transpose_a=True)

        # Compute the bound
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Luinv_c))
        bound += -0.5 * tf.reduce_sum(Kff_diag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))/sigma2

        return bound.numpy(), Kff_diag - tf.linalg.diag_part(Qff)
    
    def calculate_gamma(self, delta):
        """
        Calculate the difference between elbo_delta and elbo_m
        """
        X, y = self.X, self.y
        N = y.shape[0]
        mean = np.mean(y)
        squared_deviations = np.sum((y - mean)**2)
        sample_variance = squared_deviations / (N - 1)

        if N < 5000: 
            elbo_max = self.calculate_elbo(X)[0]
        else :
            ind = np.random.choice(N, 5000, replace=False)
            elbo_max = self.calculate_elbo(X[ind])[0]

        log_lik_joint = -0.5 * N * np.log(2 * np.pi * sample_variance) - 0.5 * squared_deviations/sample_variance

        gamma = elbo_max - delta * np.abs(elbo_max - log_lik_joint)

        return gamma
    
    def get_inducing_points(self, delta):

        X, y = self.X, self.y
        count = y.shape[0]
        mean = np.mean(y)
        M2 = np.sum((y-mean)**2)
        params = [count, mean, M2]

        gamma = self.calculate_gamma(delta)

        # Initialize the first inducing point
        cond_var = self.kernel(X, full_cov=False)
        ind = []
        ind.append(np.argmax(cond_var))

        for i in range(X.shape[0]):
            # Select the next inducing point
            Z = X[ind]

            elbo_current, cond_var = self.calculate_elbo(Z)
            # If the difference is less than gamma, stop the algorithm
            if elbo_current >= gamma:
                print("Terminating selection of inducing points, M = ", Z.shape[0])
                break
            # Else, add the point with the largest conditional variance 
            else:
                ind.append(np.argmax(cond_var))

        return Z, params
    
        
    