import numpy as np
from numba import prange, njit
from numba.typed import List
from numba.core import types

__all__ = ['dtw_distance']


@njit(parallel=True, nogil=True)
def dtw_distance(dataset):
    """
    Computes the dataset DTW distance matrix using multiprocessing.
    Args:
        dataset: timeseries dataset of shape [N1, T1]
    Returns:
        Distance matrix of shape [N1, N1]
    """
    n = dataset.shape[0]
    dist = np.empty((n, n), dtype=np.float64)

    for i in prange(n):
        for j in prange(i, n):
            distance = _dtw_distance(dataset[i], dataset[j])
            dist[i][j] = distance
            dist[j][i] = distance

    return dist


@njit(parallel=True, nogil=True)
def ngram_distance(dataset):
    """
    Computes the dataset ngram distance matrix using multiprocessing.
    Args:
        dataset: timeseries dataset of shape [N1, T1]
    Returns:
        Distance matrix of shape [N1, N1]
    """
    n = len(dataset)
    dist = np.empty((n, n), dtype=np.float64)

    for i in prange(n):
        for j in prange(i, n):
            distance = _ngram_distance(dataset[i], dataset[j])
            dist[i][j] = distance
            dist[j][i] = distance

    return dist


@njit(cache=True)
def _ngram_distance(i, j):
    i_keys = list(i.keys())
    j_keys = list(j.keys())

    ngram_all = set()
    for key in i_keys:
        ngram_all.add(key)
    for key in j_keys:
        ngram_all.add(key)

    i_vec_alt = List()
    j_vec_alt = List()
    for item in ngram_all:
        if item in i_keys:
            i_vec_alt.append(i[item])
        else:
            i_vec_alt.append(0)

        if item in j_keys:
            j_vec_alt.append(j[item])
        else:
            j_vec_alt.append(0)

    return _cosine_dist(i_vec_alt, j_vec_alt)


@njit(cache=True, fastmath=True)
def _cosine_dist(u, v, w=None):
    """
    :purpose:
    Computes the cosine similarity between two 1D arrays
    Unlike scipy's cosine distance, this returns similarity, which is 1 - distance

    :params:
    u, v   : input arrays, both of shape (n,)
    w      : weights at each index of u and v. array of shape (n,)
             if no w is set, it is initialized as an array of ones
             such that it will have no impact on the output

    :returns:
    cosine  : float, the cosine similarity between u and v
    """

    n = len(u)
    w = _init_w(w, n)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return 1 - num / denom


@njit(fastmath=True)
def _init_w(w, n):
    """
    :purpose:
    Initialize a weight array consistent of 1s if none is given
    This is called at the start of each function containing a w param

    :params:
    w      : a weight vector, if one was given to the initial function, else None
             NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
             convert it to np.float64 (using w.astype(np.float64)) before passing it to
             any function
    n      : the desired length of the vector of 1s (often set to len(u))

    :returns:
    w      : an array of 1s with shape (n,) if w is None, else return w un-changed
    """
    if w is None:
        return np.ones(n)
    else:
        return w


@njit(cache=True, fastmath=True)
def _dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 1-D
    timeseries numpy arrays.
    Args:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared.
    Returns:
        DTW distance between A and B
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    v = series1[0] - series2[0]
    E[0][0] = v * v

    # Fill First Column
    for i in range(1, l1):
        v = series1[i] - series2[0]
        E[i][0] = E[i - 1][0] + v * v

    # Fill First Row
    for i in range(1, l2):
        v = series1[0] - series2[i]
        E[0][i] = E[0][i - 1] + v * v

    for i in range(1, l1):
        for j in range(1, l2):
            v = series1[i] - series2[j]
            v = v * v

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])
