import warnings
from typing import Callable, Optional

import numpy as np
import scipy.cluster
import warnings

"""
Burt et al. (2020) - Convergence of Sparse Variational Inference in Gaussian Processes Regression
https://www.jmlr.org/papers/volume21/19-1015/19-1015.pdf
"""

np.seterr(all='raise')

def sample_discrete(unnormalized_probs):
    unnormalized_probs = np.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = np.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        warnings.warn("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]


class InducingPointInitializer:
    def __init__(self, seed: Optional[int] = 0, randomized: Optional[bool] = True, **kwargs):
        self._randomized = randomized
        self.seed = seed if self.randomized else None

    def __call__(self, training_inputs: np.ndarray, M: int,
                 kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        if self.seed is not None:
            restore_random_state = np.random.get_state()
            np.random.seed(self.seed)
        else:
            restore_random_state = None

        Z = self.compute_initialisation(training_inputs, M, kernel)

        if self.seed is not None:
            np.random.set_state(restore_random_state)

        return Z

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        raise NotImplementedError

    @property
    def randomized(self):
        return self._randomized

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if k not in ['_randomized']])
        return f"{type(self).__name__}({params})"

class ConditionalVariance(InducingPointInitializer):
    def __init__(self, sample: Optional[bool] = False, threshold: Optional[int] = 0.0, seed: Optional[int] = 0,
                 **kwargs):
        """
        :param sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
        point with highest variance at each iteration
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        super().__init__(seed=seed, randomized=True, **kwargs)
        self.sample = sample
        self.threshold = threshold

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        """
        The version of this code without sampling follows the Greedy approximation to MAP for DPPs in
        @incollection{NIPS2018_7805,
                title = {Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity},
                author = {Chen, Laming and Zhang, Guoxin and Zhou, Eric},
                booktitle = {Advances in Neural Information Processing Systems 31},
                year = {2018},
            }
        and the initial code is based on the implementation of this algorithm (https://github.com/laming-chen/fast-map-dpp)
        It is equivalent to running a partial pivoted Cholesky decomposition on Kff (see Figure 2 in the below ref.),
        @article{fine2001efficient,
                title={Efficient SVM training using low-rank kernel representations},
                author={Fine, Shai and Scheinberg, Katya},
                journal={Journal of Machine Learning Research},
                year={2001}
            }

        TODO: IF M ==1 this throws errors, currently throws an assertion error, but should fix
        Initializes based on variance of noiseless GP fit on inducing points currently in active set
        Complexity: O(NM) memory, O(NM^2) time
        :param training_inputs: [N,D] numpy array,
        :param M: int, number of points desired. If threshold is None actual number returned may be less than M
        :param kernel: kernelwrapper object
        :return: inducing inputs, indices,
        [M,D] np.array to use as inducing inputs,  [M], np.array of ints indices of these inputs in training data array
        """
        N = training_inputs.shape[0]
        perm = np.random.permutation(N)  # permute entries so tiebreaing is random
        training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        di = kernel(training_inputs, None, full_cov=False) + 1e-6  # jitter
        if self.sample:
            indices[0] = sample_discrete(di)
        else:
            indices[0] = np.argmax(di)  # select first point, add to index 0
        if M == 1:
            indices = indices.astype(int)
            Z = training_inputs[indices]
            indices = perm[indices]
            return Z, indices
        ci = np.zeros((M - 1, N))  # [M,N]
        for m in range(M - 1):
            j = int(indices[m])  # int
            new_Z = training_inputs[j:j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
            L = np.round(np.squeeze(Lraw), 20)  # [N]
            L[j] += 1e-6  # jitter
            ei = (L - np.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei ** 2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            if self.sample:
                indices[m + 1] = sample_discrete(di)
            else:
                indices[m + 1] = np.argmax(di)  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if np.sum(np.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                print("Stopping early")
                break
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        print("Inducing points", Z.shape[0])
        return Z, indices

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()
                            if
                            k not in ['_randomized'] and
                            not (k == "threshold" and self.threshold == 0.0)])
        return f"{type(self).__name__}({params})"
