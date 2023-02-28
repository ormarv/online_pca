import itertools

import numpy as np


def dict2numpy(value_dict, keys=None):
    # changes a dictionairy into a numpy array.
    # if keys is provided a then the order of numpy array
    # reflect the order of keys iterator
    ls = []
    if keys is None:
        for i in value_dict.keys():
            ls.append(float(value_dict[i]))
    else:
        for i in keys:
            ls.append(float(value_dict[i]))
    return np.array(ls)


def vec2dict(arr, keys=None):
    # change a numpy vector (dim=1) to a dictionairy
    # if keys is provided the keys of the ditionairy will
    # be of the same order and values as keys
    if keys is None:
        return {idx: float(i) for idx, i in enumerate(arr)}
    else:
        return {key: float(i) for (key, i) in zip(keys, arr)}


def mat2dict(arr):
    # change a numpy matrix to a dictionairy
    r, c = arr.shape
    arr_dict = {}
    for i in range(r):
        for j in range(c):
            arr_dict[(i, j)] = float(arr[i, j])
    return arr_dict


def dotvecmat(x, A):
    # a modified version of river.utils.math.dotvecmat
    C = {}

    for (i, xi), ((j, k), ai) in itertools.product(x.items(), A.items()):
        if i == j:
            C[k] = C.get(k, 0.0) + xi * ai

    return C


def svd_flip(u, v, u_based_decision=True):
    """
    THIS FUNCTION IS COPIED FROM SKLEARN.
    It was copied since river doesnot have a depency on sklearn  and it is
    not worth adding a dependency for such a small function.

    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.
    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
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
