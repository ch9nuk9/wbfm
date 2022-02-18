import logging

import numpy as np
from lmfit.models import GaussianModel, ConstantModel
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def aicc_correction(p, n):
    return 2 * p * (p + 1) / (n - p - 1)


def aicc_from_fit(result):
    aic = result.aic
    aicc = aic + aicc_correction(result.nvarys, len(result.data))
    return aicc


def get_best_model_using_aicc(list_of_models):
    all_aicc = [aicc_from_fit(model) for model in list_of_models]
    return np.argmin(all_aicc)


def plot_gaussians(result, split_point):
    y = result.data
    x, y = np.arange(len(y)), np.array(y)
    peak1 = result.values['g1_center']
    peak2 = result.values['g2_center']

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y)
    # axes[0].plot(x, init, '--', label='initial fit')
    axes[0].plot(x, result.best_fit, '-', label='best fit')
    axes[0].legend()

    comps = result.eval_components(x=x)
    axes[1].plot(x, y)
    axes[1].plot(x, comps['g1_'], '--', label='Gaussian component 1')
    axes[1].plot(peak1, y[int(peak1)], 'ro', label='Peak of gaussian 1')
    if 'g2_' in comps:
        axes[1].plot(x, comps['g2_'], '--', label='Gaussian component 2')
        axes[1].set_title("Best fit is two gaussians")
        axes[1].plot(peak2, y[int(peak2)], 'ro', label='Peak of gaussian 2')
    else:
        axes[1].set_title("Best fit is one gaussian")

    if split_point:
        axes[1].plot([split_point, split_point], [0, np.max(y)], 'k', label='Split line')
    axes[1].legend()

    plt.show()


def calculate_multi_gaussian_fits(y, min_separation, background):
    """Calculates a 1 and 2 gaussian fit, with the goal of using aicc to pick the best option

    Note: min_separation is just used to initialize the gaussian widths, and is not a threshold
    """
    x, y = np.arange(len(y)), np.array(y)

    y -= background

    # 2 gaussians
    gauss1 = GaussianModel(prefix='g1_')
    pars = gauss1.make_params()

    pars['g1_center'].set(value=len(y) / 4.0, min=0, max=len(y))
    pars['g1_sigma'].set(value=min_separation / 2, min=1, max=3)
    pars['g1_amplitude'].set(value=np.mean(y), min=0)

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())

    pars['g2_center'].set(value=3 * len(y) / 4.0, min=0, max=len(y))
    pars['g2_sigma'].set(value=min_separation / 2, min=1, max=3)
    pars['g2_amplitude'].set(value=np.mean(y), min=0)

    mod = gauss1 + gauss2

    out = mod.fit(y, pars, x=x)

    plt.show()

    results_2gauss = out

    # 1 gaussian
    gauss1 = GaussianModel(prefix='g1_')
    pars = gauss1.make_params()
    pars['g1_center'].set(value=len(y) / 2.0, min=0, max=len(y))
    pars['g1_sigma'].set(value=min_separation / 2, min=1, max=3)
    pars['g1_amplitude'].set(value=np.mean(y), min=0)

    mod = gauss1

    out = mod.fit(y, pars, x=x)

    results_1gauss = out

    return [results_1gauss, results_2gauss]


def calc_split_point_from_gaussians(result):
    """Must be determined elsewhere if two are really there"""
    y = result.data
    x = np.arange(len(y))

    g1 = result.eval_components(x=np.array(x))['g1_']
    g2 = result.eval_components(x=np.array(x))['g2_']
    diff = np.array(np.abs(g1 - g2))

    peak1 = result.values['g1_center']
    peak2 = result.values['g2_center']
    if peak1 > peak2:
        peak1, peak2 = peak2, peak1
    # print(peak1, peak2)
    peaks_of_gaussians = [int(np.floor(peak1)), int(np.ceil(peak2))]

    ind = peaks_of_gaussians[0] + np.array(range(0, peaks_of_gaussians[1] + 1))
    ind = np.clip(ind, 0, len(diff)-1)
    inter_peak_diff = np.array(diff[ind])

    try:
        split_point = find_peaks(-inter_peak_diff)[0][0] + ind[0]
    except IndexError:
        logging.warning("Could not split")
        split_point = None

    # Then give the middle point to the smallest neuron
    len1, len2 = len(y[:split_point + 1]), len(y[(split_point+1):])
    if len1 < len2:
        split_point += 1
    if len1 > len2:
        split_point -= 1
    else:
        # If tie, just leave it on the left
        pass

    return split_point


def OLD_calc_split_point_from_gaussians(peaks_of_gaussians, y_data):
    if peaks_of_gaussians is None:
        return None
    # Plan a: find the peak between the gaussian blobs
    inter_peak_brightnesses = np.array(y_data[peaks_of_gaussians[0] + 1:peaks_of_gaussians[1]])
    split_point, _ = find_peaks(-inter_peak_brightnesses)
    if len(split_point) > 0:
        split_point = int(split_point[0])
        split_point += peaks_of_gaussians[0] + 1
    else:
        # Plan b: Just take the average
        split_point = int(np.mean(peaks_of_gaussians)) + 1
    return split_point
