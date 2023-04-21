import numpy as np
from scipy.stats._resampling import _permutation_test_iv, PermutationTestResult


def custom_permutation_test(data, statistic, *, permutation_type='stratified',
                     vectorized=False, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, random_state=None):
    """
    Same as scipy.stats.permutation_test, but adds a permutation_type:
        'stratified' - permutes larger subsets of data at once. For example, if many samples are from the same
            individual, this will permute all samples from that individual at once

    Parameters
    ----------
    data
    statistic
    permutation_type
    vectorized
    n_resamples
    batch
    alternative
    axis
    random_state

    Returns
    -------

    """
    raise NotImplementedError
    args = _permutation_test_iv(data, statistic, permutation_type, vectorized,
                                n_resamples, batch, alternative, axis,
                                random_state)
    (data, statistic, permutation_type, vectorized, n_resamples, batch,
     alternative, axis, random_state) = args

    observed = statistic(*data, axis=-1)

    null_calculator_args = (data, statistic, n_resamples,
                            batch, random_state)
    calculate_null = _calculate_null_stratified
    null_distribution, n_resamples, exact_test = (calculate_null(*null_calculator_args))

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return PermutationTestResult(observed, pvalues, null_distribution)


def _calculate_null_stratified(data, statistic, n_permutations, batch,
                               random_state=None):
    """
    Similar to scipy.stats.resampling._calculate_null_both, but for a stratified sampling scheme

    Parameters
    ----------
    data
    statistic
    n_permutations
    batch
    random_state

    Returns
    -------

    """
    raise NotImplementedError