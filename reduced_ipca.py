import collections

import numpy as np
import scipy.linalg as linalg
from river import base

from utils import *


class ReducedPCA(base.Transformer):
    """
    An online/incremental PCA algorithm based on the pertubation
    Parameters
    ----------

    n_components : int
        The number of components to keep.

    ff : float, default=0.1
        Forgetting factor, not implemented



    References
    ----------
    ARORA, Raman, COTTER, Andrew, LIVESCU, Karen, et al. Stochastic optimization for PCA and PLS. In
    : 2012 50th annual allerton conference on communication, control, and computing (allerton).
    IEEE, 2012. p. 861-868.

    Examples
    --------
    >>> from reduced_ipca import ReducedPCA
    >>> from utils import dict2numpy, vec2dict
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> from sklearn.decomposition import PCA
    >>> test_data = np.random.normal(size=(10_000,10))*np.arange(10)
    >>> test_data = [{idx: i  for idx, i in enumerate(test_data[i])} for i in range(10_000)]
    >>> test_vec = {idx: i  for idx, i in enumerate(np.arange(10))}
    >>> ipca = ReducedPCA(5)
    >>> for i in test_data:
    >>>     ipca.learn_one(i)
    >>> print(ipca.inverse_transform_one(ipca.transform_one(test_vec)))
    {0: 0.0, 1: 0.025599074602075535, 2: 0.01266575746956531, 3: 0.12288722014933189,
    4: 0.2844437990407748, 5: 5.18922495580125, 6: 5.970517303933131,
    7: 7.023804221989852, 8: 7.992712204209254, 9: 9.046572082523122}
    """

    def __init__(
        self,
        n_components,
        ff=1.0,
    ):
        self.n_components = n_components
        self.counter = 0
        self.ff = ff
        self.np_mean = None
        self.dict_components = None
        self.np_singular_val = None
        self.np_components = None
        self.sample = None

    def transform_one(self, X):
        """
        Apply dimensionality reduction to one data point.
        -------------

        Parameters
        ----------

        X : a data point on which to apply dimensionality reduction.

        Returns
        --------

        A dictionary of length n_components with the principal components values of X.
        """
        tmp_sample = {}
        for idx, k in enumerate(self.sample.keys()):
            tmp_sample[idx] = X[k] - self.mean[idx]
        return dotvecmat(tmp_sample, self.dict_components)

    def inverse_transform_one(self, X):
        """
        Gets a compressed version of the original data from the principal components of one datapoint.

        Parameters
        -----------
            X : the result of transform_one.

        Returns
        -------
            A dictionary with the same size and keys as the original input data.
        """
        tmp_result = dotvecmat(X, self.dict_inverse_components)
        to_return = {}
        for idx, k in enumerate(self.sample.keys()):
            to_return[k] = tmp_result[idx] + self.mean[idx]
        return to_return

    def _start(self, X):
        # internal function to intialize the variables.
        self.sample = collections.OrderedDict()
        for k, v in X.items():
            self.sample[k] = v
        self.vec_sample = dict2numpy(self.sample)
        self.mean = self.vec_sample
        self.np_components = np.zeros((self.n_components, len(self.sample)))
        self.np_singular_val = np.zeros((self.n_components,))
        self.counter = 1
        return

    def learn_one(self, X):
        """
        Fits the model on one data point.

        Parameters
        ------------
        X:  a data point for the model to learn
        """
        if self.sample is None:
            self._start(X)
        else:
            for key in self.sample.keys():
                self.sample[key] = X[key]
            self.vec_sample = dict2numpy(self.sample)
            self.vec_sample -= self.mean
            c_n1 = self.np_components @ self.vec_sample
            x_perp = self.vec_sample - self.np_components.T @ c_n1
            x_perp_norm = np.linalg.norm(x_perp)

            upper_matrix = np.hstack(
                [
                    (self.counter + 1) * np.diag(self.np_singular_val)
                    + np.outer(c_n1, c_n1),
                    (x_perp_norm * c_n1).reshape(-1, 1),
                ]
            )

            lower_matrix = np.hstack([x_perp_norm * c_n1, x_perp_norm**2])

            q_n1 = (self.counter / (self.counter + 1) ** 2) * np.vstack(
                [upper_matrix, lower_matrix.reshape(1, -1)]
            )

            U, S, Vt = linalg.svd(q_n1, full_matrices=False, check_finite=False)
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

            self.np_components = (
                self.ff
                * Vt[:-1]
                @ np.vstack(
                    [self.np_components, (1 / x_perp_norm) * x_perp.reshape(1, -1)]
                )
                + (1 - self.ff) * self.np_components
            )

            self.np_singular_val = (
                self.ff * S[:-1] + (1 - self.ff) * self.np_singular_val
            )
            # print(self.np_singular_val )
            self.mean = (self.counter / (self.counter + 1)) * self.mean + (
                1 / (self.counter + 1)
            ) * self.vec_sample
            self.counter += 1
        self.dict_components = mat2dict(self.np_components.T)
        self.dict_inverse_components = mat2dict(self.np_components)
