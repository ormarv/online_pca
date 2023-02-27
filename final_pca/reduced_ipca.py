import collections 
import numpy as np
import scipy.linalg as linalg
from river import base
from utils import *

class ReducedPCA(base.Transformer):
    #TODO add base
    """ 
        #TODO add documentation 
        Very good documentation should be added.      
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
        tmp_sample = {}
        for idx, k in enumerate(self.sample.keys()):
            tmp_sample[idx]  = X[k]-self.mean[idx] 
        return dotvecmat(tmp_sample, self.dict_components)


    def inverse_transform_one(self,X):
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
        if self.sample==None:
            self._start(X)
        else: 
            for key in self.sample.keys():
                self.sample[key] = X[key]
            self.vec_sample = dict2numpy(self.sample)
            self.vec_sample -= self.mean
            c_n1 = self.np_components @ self.vec_sample
            x_perp = self.vec_sample - self.np_components.T@c_n1 
            x_perp_norm = np.linalg.norm(x_perp)
            
            upper_matrix = np.hstack([(self.counter+1)*np.diag(self.np_singular_val)+np.outer(c_n1, c_n1),
                                    (x_perp_norm*c_n1).reshape(-1, 1)])
            
            lower_matrix = np.hstack([x_perp_norm*c_n1, x_perp_norm**2])

            q_n1 = (self.counter/(self.counter+1)**2)*np.vstack([upper_matrix,
                             lower_matrix.reshape(1,-1)])
            
            U, S, Vt = linalg.svd(q_n1, full_matrices=False, check_finite=False)
            if self.deterministic:
                U, Vt = svd_flip(U, Vt, u_based_decision=False)
            
            
            self.np_components = Vt[:-1]@np.vstack([self.np_components,
                                            (1/x_perp_norm)*x_perp.reshape(1,-1)])

            self.np_singular_val =  S[:-1]
            # print(self.np_singular_val )
            self.mean = (self.counter/(self.counter+1))*self.mean + (
                1/(self.counter+1))*self.vec_sample
            self.counter+=1
        self.dict_components = mat2dict(self.np_components.T)
        self.dict_inverse_components = mat2dict(self.np_components)
