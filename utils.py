
import numpy as np 
import itertools
#TODO add documentation 
def dict2numpy(value_dict, keys=None):
    ls = []
    if keys==None:
        for i in value_dict.keys():
            ls.append(float(value_dict[i]))
    else:
        for i in keys:
            ls.append(float(value_dict[i]))
    return np.array(ls)

def vec2dict(arr, keys=None):
    if keys==None:
        return {
            idx:float(i) for idx, i in enumerate(arr)
        }
    else: 
        return { key:float(i) for (key,i) in zip(keys,arr)         
        }

def mat2dict(arr):
    r, c = arr.shape
    arr_dict = {}
    for i in range(r):
        for j in range(c):
            arr_dict[(i,j)] = float(arr[i,j])
    return arr_dict


def dotvecmat(x, A):
    C = {}

    for (i, xi), ((j, k), ai) in itertools.product(x.items(), A.items()):
        if i == j:
            C[k] = C.get(k, 0.0) + xi * ai

    return C


def svd_flip(u, v, u_based_decision=True):
    #TODO add sklearn citation
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v