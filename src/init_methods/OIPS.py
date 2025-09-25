import numpy as np
import tensorflow as tf

"""
Galy-Fajou et al. (2020) - Online Inducing Point Selection for Sparse Gaussian Processes
https://arxiv.org/pdf/2107.10066.pdf
Code adapted from 
https://github.com/JuliaGaussianProcesses/InducingPoints.jl/blob/275613d9413429fe42d28bc3e90a3dc2bd491318/src/online/oips.jl
and the paper's algorithm description.
"""


class OIPS:
    def __init__(self, rho_accept=0.8, eta=0.95, kmax=float('inf'), kmin=10, rho_remove=float('inf')):
        self.rho_accept = rho_accept
        self.eta = eta
        self.kmax = kmax
        self.kmin = kmin
        self.rho_remove = np.sqrt(rho_accept) if np.isinf(rho_remove) else rho_remove
        
        if not 0 < self.rho_accept < 1:
            raise ValueError("rho_accept should be between 0 and 1")
        if not 0 < self.eta < 1:
            raise ValueError("eta should be between 0 and 1")
        if rho_remove == float('inf'):
            self.rho_remove = np.sqrt(rho_accept)
        if not 0 < self.rho_remove < 1:
            raise ValueError("rho_remove should be between 0 and 1")
        
    def __repr__(self):
        return f"Online Inducing Point Selection (rho_accept: {self.rho_accept}, rho_remove: {self.rho_remove}, kmax: {self.kmax})"
    
    def select_inducing_points(self, X, kernel):
        N = X.shape[0]
        if N < self.kmin:
            raise ValueError(f"First batch should have at least {self.kmin} samples")
        cond_var = kernel(X, None, full_cov=False)
        # random
        idx = tf.random.shuffle(tf.range(N))[:self.kmin]
        Z = X[idx]
        return self.update_inducing_points(Z, X, kernel)

    def update_inducing_points(self, Z, X, kernel):
        if hasattr(kernel, 'kernels'):
            kernel = kernel.kernels[1] ## For magnetometer experiments
        rho_accept = kernel.variance.numpy() * self.rho_accept
        for x in X:
            x = x.reshape(1, -1)
            kx = kernel.K(x, Z) 
            if tf.reduce_max(kx) < rho_accept:
                Z = tf.concat([Z, x], axis=0)
            while Z.shape[0] > self.kmax:
                K = kernel.K(Z, Z)
                m = tf.reduce_max(K - tf.linalg.diag(tf.linalg.diag_part(K)))
                self.rho_remove = self.eta * m
                Z = self.remove_point(Z, K)
                if self.rho_remove < self.rho_accept:
                    self.rho_accept = self.eta * self.rho_remove
                print(f"rho_accept reset to: {self.rho_accept}")
        
        print(f"Final number of inducing points: {Z.shape[0]}")
        return Z
    
    def remove_point(self, Z, K):
        if Z.shape[0] > self.kmin:
            overlap_count = tf.reduce_sum(tf.cast(K > self.rho_remove, tf.int32), axis=1)
            removable = tf.where(overlap_count > 1)[:, 0]
            
            while tf.size(removable) > 0 and Z.shape[0] > self.kmin:
                i = tf.random.shuffle(removable)[0]
                connected = tf.where(K[i] > self.rho_remove)[:, 0]
                overlap_count = tf.tensor_scatter_nd_sub(overlap_count, connected[:, tf.newaxis], tf.ones_like(connected))
                removable = tf.boolean_mask(removable, tf.gather(overlap_count, removable) > 1)
                Z = tf.concat([Z[:i], Z[i+1:]], axis=0)
                K = tf.concat([tf.concat([K[:i], K[i+1:]], axis=0)[:, :i], 
                               tf.concat([K[:i], K[i+1:]], axis=0)[:, i+1:]], axis=1)
        
        return Z