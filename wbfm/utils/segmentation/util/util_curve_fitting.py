import logging

import numpy as np
from lmfit.models import GaussianModel, ConstantModel
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def aicc_correction(p, n):
    numer = (n - p - 1)
    if numer > 0:
        return 2 * p * (p + 1) / numer
    elif numer <= 0:
        return np.inf


def aicc_from_fit(result):
    aic = result.aic
    aicc = aic + aicc_correction(result.nvarys, len(result.data))
    return aicc


def get_best_model_using_aicc(list_of_models):
    all_aicc = [aicc_from_fit(model) for model in list_of_models]
    return np.argmin(all_aicc), all_aicc


def plot_gaussians(result, split_point, prefixes=None):
    if prefixes is None:
        prefixes = ['g1_', 'g2_', 'g3_']
    y = result.data
    x, y = np.arange(len(y)), np.array(y)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y)
    # axes[0].plot(x, init, '--', label='initial fit')
    axes[0].plot(x, result.best_fit, '-', label='best fit')
    axes[0].legend()

    comps = result.eval_components(x=x)
    axes[1].plot(x, y)
    for i, prefix in enumerate(prefixes):
        if prefix in comps:
            axes[1].plot(x, comps[prefix], '--', label=f'Gaussian component {i+1}')
            peak1 = result.values[f'{prefix}center']
            axes[1].plot(peak1, y[int(peak1)], 'ro', label=f'Peak of gaussian {i+1}')
            axes[1].set_title(f"Best fit is {i+1} gaussian(s)")

    plt.ylabel('Brightness (sum of pixels in each segmented plane)')
    plt.xlabel('Z slice (starts at top of current neuron not volume)')

    if split_point:
        axes[1].plot([split_point, split_point], [0, np.max(y)], 'k', label='Split line')
    axes[1].legend()

    plt.show()


def calculate_multi_gaussian_fits(y, background, allow_3_gaussians=False):
    """Calculates a 1 and 2 gaussian fit, with the goal of using aicc to pick the best option

    Note: min_separation is just used to initialize the gaussian widths, and is not a threshold
    """
    x, y = np.arange(len(y)), np.array(y)
    background = np.min(y)
    y -= background

    # c = ConstantModel(value=14.0)
    # pars = c.make_params()

    # sigma_opt = dict(value=2.0, min=1, max=3)
    sigma_opt = dict(value=len(y) / 4.0, min=0.5, max=len(y) / 3.0)
    center_opt = dict(min=0, max=len(y))

    # 1, then 2 gaussians
    gauss1 = GaussianModel(prefix='g1_')
    # pars.update(gauss1.make_params())
    pars = gauss1.make_params()

    pars['g1_center'].set(value=len(y) / 4.0, **center_opt)
    pars['g1_sigma'].set(**sigma_opt)
    pars['g1_amplitude'].set(value=np.max(y), min=0)

    # pars['g1_amplitude'].vary = False

    mod = gauss1 #+ c
    out = mod.fit(y, pars, x=x)
    results_1gauss = out

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())

    pars['g2_center'].set(value=3 * len(y) / 4.0, **center_opt)
    pars['g2_sigma'].set(**sigma_opt)
    pars['g2_amplitude'].set(value=np.mean(y), min=0)

    mod = gauss1 + gauss2 #+ c
    out = mod.fit(y, pars, x=x)
    results_2gauss = out

    # If long enough, 3 gaussians
    if allow_3_gaussians and len(y) > 9:
        gauss3 = GaussianModel(prefix='g3_')
        pars.update(gauss3.make_params())
        pars['g3_center'].set(value=len(y) / 2.0, **center_opt)
        pars['g3_sigma'].set(**sigma_opt)
        pars['g3_amplitude'].set(value=np.mean(y), min=0)

        mod = gauss1 + gauss2 + gauss3 #+ c
        out = mod.fit(y, pars, x=x)
        results_3gauss = out

        return [results_1gauss, results_2gauss, results_3gauss]
    else:
        return [results_1gauss, results_2gauss]


def calc_split_point_from_gaussians(result, prefix1='g1_', prefix2='g2_'):
    """Must be determined elsewhere if two are really there"""
    y = result.data
    x = np.arange(len(y))

    g1 = result.eval_components(x=np.array(x))[prefix1]
    g2 = result.eval_components(x=np.array(x))[prefix2]
    diff = np.array(np.abs(g1 - g2))

    peak1 = result.values[f'{prefix1}center']
    peak2 = result.values[f'{prefix2}center']
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
        logging.warning("Could not find peak")
        # plt.plot(inter_peak_diff)
        split_point = None

    # Then give the middle point to the smallest neuron
    if split_point is not None:
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


def _plot_just_data(x_data, y_data):
    fig = plt.figure()
    plt.plot(x_data, y_data, label='Data')
    plt.ylim([np.min(y_data), np.max(y_data)])
    plt.title('Candidate split (line stays to the left neuron)')
    plt.ylabel('Brightness (sum of pixels in each segmented plane)')
    plt.xlabel('Z slice (starts at top of current neuron not volume)')
    plt.legend(loc='upper right')
    plt.xticks(x_data)
    plt.grid(True, axis='x')
    return fig
