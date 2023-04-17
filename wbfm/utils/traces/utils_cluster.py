import numpy as np


def ks_statistic(x, y, axis=0):
    """
    Designed for use with scipy.stats.permutation_test

    Vectorize, such that the axis (0) is batches
    Expects dimensions of: batch, time, samples

    Parameters
    ----------
    x
    y
    axis

    Returns
    -------

    """

    pipeline = lambda x: np.nanmedian(x, axis=-1)
    x_cumsum = pipeline(x)
    y_cumsum = pipeline(y)
    xy_diff = np.nanmax(np.abs(x_cumsum - y_cumsum), axis=-1)
    return xy_diff
