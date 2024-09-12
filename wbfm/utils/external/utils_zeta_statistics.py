import math
import warnings
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import detrend


# Functions all following the paper:
# https://elifesciences.org/articles/71969#s4


# @numba.jit(nopython=True)
def jitter_indices(triggered_average_indices: List[np.ndarray], max_jitter: int, max_len: int) -> \
        List[np.ndarray]:
    """
    Takes a list of arrays, and applies a random offset

    Also ensures no indices go outside the recording using max_len
    """

    ind_jittered = []
    for vec in triggered_average_indices:
        # Do not allow jitters that would put the window fully outside the recording
        this_max_jitter = np.min([max_len - vec[-1], max_jitter + 1])
        this_min_jitter = np.max([-vec[0], -max_jitter])
        new_vec = vec.copy() + np.random.randint(this_min_jitter, this_max_jitter)
        # new_vec = np.array([v for v in new_vec if v < max_len])
        ind_jittered.append(new_vec)
    return ind_jittered


def calculate_zeta_cumsum(mat: np.ndarray, DEBUG=False):
    """
    Calculates the cumulative sum minus a baseline. See:
    https://elifesciences.org/articles/71969#s4

    Parameters
    ----------
    mat

    Returns
    -------

    """
    # Equation 3
    with warnings.catch_warnings():
        # Empty slices don't matter
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trace_sum = np.nanmean(mat, axis=0)  # New: take a mean to remove influence of variable length subsets
    # trace_sum = np.nansum(mat, axis=0)
    # New: drop time points that are entirely nan, to make the baseline make sense
    trace_sum = trace_sum[~np.isnan(trace_sum)]
    alternate_methods = False
    if alternate_methods:
        # Alternate: detrend instead of subtracting a cumulative sum
        # delta = detrend(trace_sum)

        # Alternate: subtract mean before taking cumsum
        # Then the base is by "definition" 0, so it is already the delta
        delta = np.nancumsum(trace_sum - np.nanmean(trace_sum))

    else:
        trace_cumsum = np.nancumsum(trace_sum)
        # Equation 4
        base_cumsum = np.linspace(trace_cumsum[0], trace_cumsum[-1], num=len(trace_cumsum))
        # Equation 5
        delta = trace_cumsum - base_cumsum
    # Equation 6
    delta_corrected = delta - np.nanmean(delta)
    # New: pad with zeros to fix dropped points above
    num_to_pad = mat.shape[1] - len(delta_corrected)
    if num_to_pad > 0:
        delta_corrected = np.pad(delta_corrected, (0, num_to_pad), 'constant')

    if DEBUG:
        trace_cumsum = np.nancumsum(trace_sum)
        base_cumsum = np.linspace(trace_cumsum[0], trace_cumsum[-1], num=len(trace_cumsum))

        plt.figure(dpi=100)
        plt.plot(trace_cumsum)
        plt.plot(base_cumsum)
        plt.title("Precorrection, cumulative sums")

    return delta_corrected


# @numba.jit(nopython=True)
def calculate_p_value_from_zeta(zeta_dat: float, zetas_baseline: np.ndarray):
    """
    Uses gumbel distribution, because the zeta values are the max of the cumulative sum of the data

    Following:
    https://elifesciences.org/articles/71969#s4

    Parameters
    ----------
    zeta_dat
    zetas_baseline

    Returns
    -------

    """

    # Equation 17
    baseline_mean = np.mean(zetas_baseline)
    # Equations 16-19
    gamma = 0.577  # Euler–Mascheroni constant
    baseline_var = np.var(zetas_baseline)
    beta = np.sqrt(6 * baseline_var) / math.pi
    # Equation 20
    baseline_mode = baseline_mean - beta * gamma

    # Equation 21
    def cumulative_gumbel(x):
        return np.exp(-np.exp(- (x - baseline_mode) / beta))

    p = 1 - cumulative_gumbel(zeta_dat)

    return p
