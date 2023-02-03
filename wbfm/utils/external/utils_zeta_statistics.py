import math
from typing import List
import numpy as np


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
        new_vec = vec + np.random.randint(this_min_jitter, this_max_jitter)
        # new_vec = np.array([v for v in new_vec if v < max_len])
        ind_jittered.append(new_vec)
    return ind_jittered


def calculate_zeta_cumsum(mat: np.ndarray):
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
    trace_sum = np.nansum(mat, axis=0)
    trace_cumsum = np.nancumsum(trace_sum)
    # Equation 4
    base_cumsum = np.linspace(trace_cumsum[0], trace_cumsum[-1], num=len(trace_cumsum))
    # Equation 5
    delta = trace_cumsum - base_cumsum
    # Equation 6
    delta_corrected = delta - np.nanmean(delta)

    return delta_corrected


# @numba.jit(nopython=True)
def calculate_p_value_from_zeta(zeta_dat, zetas_baseline):
    """
    Uses gumbel distribution

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
