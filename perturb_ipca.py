import collections

import numpy as np
import scipy.linalg as linalg
from river import base

from utils import *


class PerturbPCA(base.Transformer):
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
    D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77,
    Issue 1-3, pp. 125-141, May 2008.

    Examples
    --------
    >>> from perturb_ipca import PerturbPCA
    >>> from utils import dict2numpy, vec2dict
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> from sklearn.decomposition import PCA
    >>> test_data = np.random.normal(size=(10_000,10))*np.arange(10)
    >>> test_data = [{idx: i  for idx, i in enumerate(test_data[i])} for i in range(10_000)]
    >>> test_vec = {idx: i  for idx, i in enumerate(np.arange(10))}
    >>> ipca = PerturbPCA(5)
    >>> for i in test_data:
    >>>     ipca.learn_one(i)
    >>> print(ipca.inverse_transform_one(ipca.transform_one(test_vec)))
    {0: -4.989979807398463e-14, 1: 0.025878167476031198, 2: 0.011633957493573332, 3: 0.12293468327409406,
    4: 0.1290879594082788, 5: 5.095516053892939, 6: 5.954303450784965,
    7: 7.0270524941953685, 8: 7.995053166631812, 9: 9.044772715368902}

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
        -------------
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
        Description :
        Fits the model on one data point.
        ------------
        Parameters :
        X : Mapping[Hashable:Any] a data point for the model to learn
        """
        if self.sample is None:
            self._start(X)
        else:
            for key in self.sample.keys():
                self.sample[key] = X[key]
            self.vec_sample = dict2numpy(self.sample)
            self.vec_sample -= self.mean
            m = self.mean.shape[0]

            mean_correction = np.sqrt((self.counter * m) / (self.counter + m)) * (
                self.mean - self.vec_sample
            )
            dec = np.vstack(
                (
                    self.np_singular_val.reshape((-1, 1)) * self.np_components,
                    self.vec_sample,
                    mean_correction,
                )
            )

            U, S, Vt = linalg.svd(dec, full_matrices=False, check_finite=False)
            U, Vt = svd_flip(U, Vt, u_based_decision=False)
            self.np_components = (1 - self.ff) * self.np_components + self.ff * Vt[
                : self.n_components
            ]
            self.np_singular_val = (1 - self.ff) * self.np_singular_val + self.ff * S[
                : self.n_components
            ]
            new_mean = (self.counter / (self.counter + 1)) * self.mean + (
                1 / (self.counter + 1)
            ) * self.vec_sample
            self.mean = self.ff * new_mean + (1 - self.ff) * self.mean
            self.counter += 1
        self.dict_components = mat2dict(self.np_components.T)
        self.dict_inverse_components = mat2dict(self.np_components)
