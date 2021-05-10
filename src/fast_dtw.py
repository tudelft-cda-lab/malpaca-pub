import numpy as np
from numba import jit, prange, njit

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
