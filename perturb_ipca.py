import collections 
import numpy as np
import scipy.linalg as linalg
from river import base
from utils import *
        
class PerturbPCA(base.Transformer):
    """ 
       
    Parameters :
    n_components : int
        The number of components to keep.
    deterministic : boolean, default=True
        Ensure deterministic 
    ff : float, default=0.1
        Forgetting factor, not implemented 
    ---------
    
    Examples :
    from reduced_ipca import ReducedPCA
    from perturb_ipca import PerturbPCA
    from utils import dict2numpy, vec2dict
    import numpy as np
    from sklearn.decomposition import PCA

    test_data = np.random.normal(size=(10_000,10))*np.arange(10)
    test_data = [{idx: i  for idx, i in enumerate(test_data[i])} for i in range(10_000)]
    test_vec = {idx: i  for idx, i in enumerate(np.arange(10))} 
    ipca = PerturbPCA(5)
    for i in test_data:
        ipca.learn_one(i)
    print(ipca.inverse_transform_one(ipca.transform_one(test_vec)))

    --------
    References :
    D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77,
    Issue 1-3, pp. 125-141, May 2008.
    """
    def __init__(self,
                 n_components,
                 ff=0.1,
                 deterministic=True):
        
        self.n_components = n_components
        self.deterministic = deterministic
        self.counter = 0 
        self.ff = ff
        self.np_mean = None        
        self.dict_components = None 
        self.np_singular_val = None
        self.np_components= None
        self.sample = None
        
    def transform_one(self,X):
        """
        Description :
        Apply dimensionality reduction to one data point.
        -------------
        Parameters :
        X :Mapping[Hashable:Any], a data point on which to apply dimensionality reduction.
        -------------
        Returns :
        A dictionary of length n_components with the principal components values of X. 
        """
        tmp_sample = {}
        for idx, k in enumerate(self.sample.keys()):
            tmp_sample[idx]  = X[k]-self.mean[idx] 
        return dotvecmat(tmp_sample, self.dict_components)


    def inverse_transform_one(self,X):
        """
        Description :
            Gets a compressed version of the original data from the principal components of one datapoint.
        # -------------
        # Parameters :
        #     X :Mapping[Hashable:Any], the result of transform_one.
        # -------------
        # Returns :
        #     A dictionary with the same size as the input data.
        """
        tmp_result = dotvecmat(X, self.dict_inverse_components)
        to_return = {}
        for idx, k in enumerate(self.sample.keys()):
            to_return[k] = tmp_result[idx]+self.mean[idx]
        return to_return

    
    def _start(self,X):
        self.sample = collections.OrderedDict()
        for k,v in X.items():
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
        if self.sample==None:
            self._start(X)
        else: 
            for key in self.sample.keys():
                self.sample[key] = X[key]
            self.vec_sample = dict2numpy(self.sample)
            self.vec_sample -= self.mean
            m = self.mean.shape[0]              

            mean_correction = np.sqrt((self.counter*m)/ (self.counter+m))*(self.mean - self.vec_sample)
            dec = np.vstack(
                (self.np_singular_val.reshape((-1,1))*self.np_components, 
                 self.vec_sample,
                 mean_correction)
            )

            U, S, Vt = linalg.svd(dec, full_matrices=False, check_finite=False)
            if self.deterministic:
                U, Vt = svd_flip(U, Vt, u_based_decision=False)
            self.np_components = Vt[:self.n_components]
            self.np_singular_val = S[:self.n_components]
            self.mean = (self.counter/(self.counter+1))*self.mean + (
                1/(self.counter+1))*self.vec_sample
            self.counter+=1
        self.dict_components = mat2dict(self.np_components.T)
        self.dict_inverse_components = mat2dict(self.np_components)       
