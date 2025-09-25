import numpy as np
import tensorflow as tf
import gpflow

from gpflow import covariances

def upper_bound(model):
    Ma = gpflow.utilities.to_default_float(tf.shape(model.Z_old)[0])
    Mb = gpflow.utilities.to_default_float(model.inducing_variable.num_inducing)
    sigma2 = model.likelihood.variance
    N = model.num_data

    Saa = model.Su_old
    ma = model.mu_old

    X = model.X
    y = model.Y
    
    # jitter
    jitter = gpflow.utilities.to_default_float(1e-12)
    
    Kff_diag = model.kernel(X, full_cov=False)
    Kbf = covariances.Kuf(model.inducing_variable, model.kernel, model.X)
    Kbb = covariances.Kuu(model.inducing_variable, model.kernel, jitter=jitter)
    Kba = covariances.Kuf(model.inducing_variable, model.kernel, model.Z_old)
    Kaa_cur = gpflow.utilities.add_noise_cov(model.kernel(model.Z_old), jitter)
    Kaa = gpflow.utilities.add_noise_cov(model.Kaa_old, jitter)

    # Cholesky decomposition
    Lb = tf.linalg.cholesky(Kbb)
    LSa = tf.linalg.cholesky(Saa)
    LKa = tf.linalg.cholesky(Kaa)

    # D 
    Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True) 
    Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
    Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
    
    d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)

    LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa, Kab_Lbinv, lower=True)
    d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

    LKainv_Kab_Lbinv = tf.linalg.triangular_solve(LKa, Kab_Lbinv, lower=True)
    d3 = tf.matmul(LKainv_Kab_Lbinv, LKainv_Kab_Lbinv, transpose_a=True)

    D = tf.eye(Mb, dtype=gpflow.default_float()) + d1/sigma2 + d2 - d3
    D = gpflow.utilities.add_noise_cov(D, jitter)
    LD = tf.linalg.cholesky(D)

    # c    
    Sainv_ma = tf.linalg.cholesky_solve(LSa, ma)
    c1 = tf.matmul(Kbf, y)
    c2 = tf.matmul(Kba, Sainv_ma)

    # Trace term t
    Qff_trace = tf.reduce_sum(tf.linalg.diag_part(tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_a=True)))
    Qaa_trace = tf.reduce_sum(tf.linalg.diag_part(tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)))
    Kaa_trace = tf.reduce_sum(tf.linalg.diag_part(Kaa_cur))
    Kff_trace = tf.reduce_sum(Kff_diag)
    t = Kff_trace + Kaa_trace - Qff_trace - Qaa_trace

    # Noise matrix with entries [[sigma2 + t, 0], [0, Da + t]]
    sigma2_hat = sigma2 + t
    
    # Da inverse
    Sainv = tf.linalg.cholesky_solve(LSa, tf.eye(Ma, dtype=gpflow.default_float()))
    Kainv = tf.linalg.cholesky_solve(LKa, tf.eye(Ma, dtype=gpflow.default_float()))
    Dainv = Sainv - Kainv

    # Da inverse and trace term
    Dainv_hat = t*Dainv + tf.eye(Ma, dtype=gpflow.default_float())
    inverse_term = tf.matmul(Dainv, tf.linalg.solve(Dainv_hat, Dainv))

    # D_hat
    LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa, Kab_Lbinv, lower=True)
    LKainv_Kab_Lbinv = tf.linalg.triangular_solve(LKa, Kab_Lbinv, lower=True)
    
    d4_hat = tf.matmul(Lbinv_Kba, tf.matmul(inverse_term, Kab_Lbinv))

    D_hat =  tf.eye(Mb, dtype=gpflow.default_float()) + d1/sigma2_hat + d2 - d3 - t*d4_hat
    D_hat = gpflow.utilities.add_noise_cov(D_hat, jitter) 
    LD_hat = tf.linalg.cholesky(D_hat)

    Dainv_hat_Sainv_ma = tf.linalg.solve(Dainv_hat, Sainv_ma)

    # c_hat = Kbf Sigma_c^{-1} y
    Dainv_hat_Sainv_ma = tf.linalg.solve(Dainv_hat, Sainv_ma)
    temp = tf.matmul(Dainv, Dainv_hat_Sainv_ma)
    c3_hat = tf.matmul(Kba, temp)
    c_hat = c1/sigma2_hat + c2 - t*c3_hat

    # Lbinv_c_hat = L_b^{-1} c_hat
    Lbinv_c_hat = tf.linalg.triangular_solve(Lb, c_hat, lower=True)

    # LDinv_Lbinv_c_hat = LD_hat^{-1}L_b^{-1} c_hat
    LDinv_Lbinv_c_hat = tf.linalg.triangular_solve(LD_hat, Lbinv_c_hat, lower=True)

    ### Constant term
    const = - 0.5 * np.log(2 * np.pi) * N 

    ### Log determinant term
    logdet = - 0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
    logdet += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

    ### Quadratic term
    quad = -0.5 * tf.reduce_sum(tf.square(y)) / sigma2_hat
    quad += -0.5 * tf.matmul(ma, Sainv_ma, transpose_a=True)
    quad += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c_hat))
    quad += 0.5 * t * tf.matmul(Sainv_ma, Dainv_hat_Sainv_ma, transpose_a=True)
    ### Delta term
    delta = - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa))) 
    delta += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LKa)))

    return const + logdet + quad + delta
