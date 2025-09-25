import numpy as np
import tensorflow as tf
import gpflow

""" 
Online Implementation of our algorithm Vegas Inducing Point Selection (VIPS)
"""

class VIPS:
    def __init__(self, X, y, kernel, Z_old, ma, Sa, Kaa, sigma2, params):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.Z_old = Z_old
        self.ma = ma
        self.Sa = Sa
        self.Kaa = Kaa
        self.sigma2 = sigma2
        self.params = params
  
        # Compute common terms
        self.common_terms = self.compute_common_terms()

        # Calculate the initial elbo
        self.elbo_max = self.calculate_elbo_max()

    def calculate_elbo_max(self):

        X, y = self.X, self.y
        Z_old = self.Z_old
        kernel = self.kernel
        N = X.shape[0]

        if N < 5000: 
            return self.calculate_elbo(tf.concat([self.X, self.Z_old], axis=0))[0]
        else :
            Nnew = 5000 - Z_old.shape[0]
            ind = np.random.choice(N, Nnew, replace=False)
            return self.calculate_elbo(tf.concat([self.X[ind], self.Z_old], axis=0))[0]

    def compute_common_terms(self):
        """
        Precompute common terms
        """
        ma = self.ma
        Saa = self.Sa
        Kaa = self.Kaa
        sigma2 = self.sigma2

        jitter = gpflow.utilities.to_default_float(1e-6)
        Kfdiag = self.kernel(self.X, full_cov=False)
        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old), jitter)
        Kaa = gpflow.utilities.add_noise_cov(Kaa, jitter)

        LSa = tf.linalg.cholesky(Saa)
        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

        Sainv_ma = tf.linalg.solve(Saa, ma)
        Sinv_y = self.y / sigma2

        return (Kfdiag, Kaa_cur, Kaa, Sinv_y, Sainv_ma, LSa, Lainv_ma)


    def calculate_elbo(self, Z):
        """
        Calculate online elbo
        """
        X, y = self.X, self.y
        Z_old = self.Z_old
        N = X.shape[0]
        Mb = Z.shape[0]
        Saa = self.Sa
        kernel = self.kernel 
        sigma2 = self.sigma2

        (Kfdiag, Kaa_cur, Kaa, Sinv_y, Sainv_ma, LSa, Lainv_ma) = self.common_terms

        jitter = gpflow.utilities.to_default_float(1e-6)

        Kbb = kernel(Z) + jitter * tf.eye(Z.shape[0], dtype=gpflow.default_float())
        Kbf = kernel(Z, X)
        Kba = kernel(Z, Z_old)


        c1 = tf.matmul(Kbf, Sinv_y)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2

        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_c = tf.linalg.triangular_solve(Lb, c, lower=True)
        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True)
        d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)/sigma2

        LSa = tf.linalg.cholesky(Saa)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(
            LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        La = tf.linalg.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(
            La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)
        Qff = tf.linalg.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_a=True)

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(y)) / sigma2
        # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # log det term
        bound += -0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))/sigma2

        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff))


        return bound, Kfdiag - tf.linalg.diag_part(Qff)

    def calculate_gamma(self,  delta):
        """
        Calculate the gamma value
        """
        X, y = self.X, self.y
        params = self.params
        Nnew = y.shape[0]
        mu_new = np.mean(y)
        count, mean, M2 = params
        Nold = count
        count += Nnew
        prev_mean = mean
        mu_new = np.mean(y)
        # Welford's online mean algorithm
        mean += (mu_new - mean)* Nnew / count
        # Chan et al parallel variance update
        delta_mean = (mu_new - prev_mean)**2
        M2_new = np.sum((y-mu_new)**2)
        M2 += M2_new + Nold*Nnew*delta_mean/count
        sample_variance = M2 / (count - 1)

        # Log likelihood is the log marginal of a Gaussian with mean and sample variance 
        log_lik_joint = -0.5 * Nnew * np.log(2 * np.pi * sample_variance) - 0.5 * np.sum((y - mean)**2 )/sample_variance

        # Calculate the gamma value
        gamma = self.elbo_max - delta * np.abs(self.elbo_max - log_lik_joint)
        return gamma, [count, mean, M2]
    
    def update_inducing_points(self, delta = 0.05):
        """
        Add inducing points to the set Z based on the difference between lower bound and upper bound
        """
        X, y = self.X, self.y
        Z_old = self.Z_old
        kernel = self.kernel
        N = X.shape[0]

        gamma, params = self.calculate_gamma(delta)

            
        elbo_current, cond_var = self.calculate_elbo(Z_old)
        if elbo_current >= gamma:
            print("Terminating selection of inducing points, M = ", Z_old.shape[0])
            return Z_old, params
        
        ind = [np.argmax(cond_var)]
        
        for _ in range(N):
            new_Z = X[ind]
            Zs = tf.concat([Z_old, new_Z], axis=0)
            
            elbo_current, cond_var = self.calculate_elbo(Zs)
            if elbo_current >= gamma:
                print("Terminating selection of inducing points, M = ", Zs.shape[0])
                break
            else:
                ind.append(np.argmax(cond_var))
        
        new_Z = X[ind]
        Zs = tf.concat([Z_old, new_Z], axis=0)
        return Zs, params
