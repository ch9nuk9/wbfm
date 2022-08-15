import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from lmfit.models import ExponentialModel
from sklearn.preprocessing import StandardScaler

from wbfm.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.general.postures.centerline_classes import shade_using_behavior
from matplotlib import transforms
from matplotlib.ticker import NullFormatter
from tqdm.auto import tqdm

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.utils_plot_traces import build_trace_factory, check_default_names, set_big_font


##
## New functions for use with project_config files
##

def make_grid_plot_using_project(project_data: ProjectData,
                                 channel_mode: str,
                                 calculation_mode: str,
                                 neuron_names_to_plot: list = None,
                                 filter_mode: str = 'no_filtering',
                                 color_using_behavior=True,
                                 remove_outliers=False,
                                 to_save=True):
    """

    See project_data.calculate_traces for details on the arguments, and TracePlotter for even more detail

    Parameters
    ----------
    project_data
    channel_mode
    calculation_mode
    neuron_names_to_plot
    filter_mode
    color_using_behavior
    remove_outliers
    to_save

    Returns
    -------

    """
    if channel_mode == 'all':
        all_modes = ['red', 'green', 'ratio', 'linear_model']
        opt = dict(project_data=project_data,
                   calculation_mode=calculation_mode,
                   color_using_behavior=color_using_behavior)
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        # Also try to remove outliers and filter
        all_modes = ['ratio', 'linear_model']
        opt['remove_outliers'] = True
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        opt['filter_mode'] = 'rolling_mean'
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        return
    if neuron_names_to_plot is not None:
        neuron_names = neuron_names_to_plot
    else:
        neuron_names = list(set(project_data.green_traces.columns.get_level_values(0)))
    # Guess a good shape for subplots
    neuron_names.sort()

    # Build functions to make a single subplot
    options = {'channel_mode': channel_mode, 'calculation_mode': calculation_mode, 'filter_mode': filter_mode,
               'remove_outliers': remove_outliers}
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = lambda axis: project_data.shade_axis_using_behavior(axis)
    logger = project_data.logger

    make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func, color_using_behavior, logger)

    # Save final figure
    if to_save:
        if neuron_names_to_plot is None:
            prefix = f"{channel_mode}_{calculation_mode}"
            if remove_outliers:
                prefix = f"{prefix}_outliers_removed"
            if filter_mode != "no_filtering":
                prefix = f"{prefix}_{filter_mode}"
            fname = f"{prefix}_grid_plot.png"
        else:
            fname = f"{len(neuron_names_to_plot)}neurons_{channel_mode}_{calculation_mode}_grid_plot.png"
        traces_cfg = project_data.project_config.get_traces_config()
        out_fname = traces_cfg.resolve_relative_path(fname, prepend_subfolder=True)

        save_grid_plot(out_fname)


def make_grid_plot_from_leifer_file(fname: str,
                                    channel_mode: str = 'all',
                                    color_using_behavior=True):
    if channel_mode == 'all':
        all_modes = ['rRaw', 'gRaw', 'Ratio2']
        opt = dict(fname=fname,
                   color_using_behavior=color_using_behavior)
        for mode in all_modes:
            make_grid_plot_from_leifer_file(channel_mode=mode, **opt)
        return

    assert channel_mode in ['rRaw', 'gRaw', 'Ratio2']

    data = scipy.io.loadmat(fname)

    ethogram = [cast_int_or_nan(d) for d in data['behavior'][0][0][0]]
    # ethogram_names = {-1: 'Reversal', 1: 'Forward', 2: 'Turn'}
    ethogram_cmap = {-1: 'darkgray', 0: None, 1: None, 2: 'red'}

    num_neurons, t = data[channel_mode].shape
    neuron_names = [int2name_neuron(i + 1) for i in range(num_neurons)]

    # Build functions to make a single subplot
    get_data_func = lambda neuron_name: (np.arange(t), data[channel_mode][name2int_neuron_and_tracklet(neuron_name) - 1])
    shade_plot_func = lambda axis: shade_using_behavior(ethogram, axis, cmap=ethogram_cmap)
    logger = logging.getLogger()

    make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func, color_using_behavior, logger)

    # Save final figure
    out_fname = f"leifer_{channel_mode}_grid_plot.png"
    out_fname = Path(fname).with_name(out_fname)

    save_grid_plot(out_fname)


def save_grid_plot(out_fname):
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.0,
                        hspace=0.0)
    logging.info(f"Saving figure at: {out_fname}")
    plt.savefig(out_fname, bbox_inches='tight', pad_inches=0)


def make_grid_plot_from_callables(get_data_func: callable,
                                  neuron_names: list,
                                  shade_plot_func: callable,
                                  color_using_behavior: bool = True,
                                  logger: logging.Logger = None):
    """

    Parameters
    ----------
    color_using_behavior - boolean for actually shading
    get_data_func - function that accepts a neuron name and returns a tuple of (t, y)
    neuron_names - list of neurons to plot
    shade_plot_func - function that accepts an axis object and shades the plot
    logger

    Example:
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = project_data.shade_axis_using_behavior

    Returns
    -------

    """
    # Loop through neurons and plot
    num_neurons = len(neuron_names)
    num_columns = 5
    num_rows = int(np.ceil(num_neurons / float(num_columns)))
    if logger is not None:
        logger.info(f"Found {num_neurons} neurons; shaping to grid of shape {(num_rows, num_columns)}")
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(25, 15), sharex=True, sharey=False)
    for ax, neuron_name in tqdm(zip(fig.axes, neuron_names), total=len(neuron_names)):

        t, y = get_data_func(neuron_name)
        ax.plot(t, y, label=neuron_name)
        # For removing the lines from the legends:
        # https://stackoverflow.com/questions/25123127/how-do-you-just-show-the-text-label-in-plot-legend-e-g-remove-a-labels-line
        leg = ax.legend(loc='upper left', handlelength=0, handletextpad=0, fancybox=True, framealpha=0.0)
        for item in leg.legendHandles:
            item.set_visible(False)
        # ax.set_title(neuron_name, {'fontsize': 28}, y=0.7)
        ax.set_frame_on(False)
        ax.set_axis_off()
        if color_using_behavior:
            shade_plot_func(ax)


##
## Functions for use with data from 'extract_all_traces'
##

def visualize_traces_with_reference(all_traces,
                                    reference_ind, reference_name,
                                    all_names=None,
                                    to_normalize=True,
                                    to_save=False):
    """
    Plot all neurons on a reference, given by reference_ind
    """
    all_names = check_default_names(all_names, len(all_traces))

    reference_trace = all_traces[reference_ind]

    for i, t_dict in enumerate(all_traces):
        if i == reference_ind:
            continue
        # Plot looped trace and reference
        ax1, ax2 = visualize_mcherry_and_gcamp(reference_trace, reference_name,
                                               make_new_title=False,
                                               to_normalize=to_normalize)
        visualize_mcherry_and_gcamp(t_dict, all_names[i],
                                    make_new_fig=False,
                                    make_new_title=False,
                                    ax1=ax1, ax2=ax2,
                                    to_normalize=to_normalize)
        if to_save:
            plt.savefig(f'traces_{all_names[i]}_ref_{reference_name}')


##
## Generally plotting
##


def plot3d_with_max(dat, z, t, max_ind, vmin=100, vmax=400):
    plt.imshow(dat[:, :, z, t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t, 1], max_ind[t, 0]
    if z == max_ind[t, 2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is on z={max_ind[t, 2]}, xy={x},{y}")


def plot3d_with_max_and_hist(dat, z, t, max_ind):
    # From: https://matplotlib.org/2.0.2/examples/pylab_examples/scatter_hist.html
    rot = transforms.Affine2D().rotate_deg(90)
    nullfmt = NullFormatter()  # no labels

    plt.figure(1, figsize=(8, 8))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axIm = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Actually display
    frame = dat[:, :, z, t]
    axIm.imshow(frame, vmin=0, vmax=400)
    x, y = max_ind[t, 1], max_ind[t, 0]
    #     if z == max_ind[t,2]:
    #         plt.scatter(x, y, marker='x', c='r')
    #     plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")

    axHistx.plot(np.max(frame, axis=0))

    #     base = plt.gca().transData
    axHisty.plot(np.flip(np.max(frame, axis=1)), range(frame.shape[0]))  # , transform=base+rot)


##
## Helper functions
##


def get_tracking_channel(t_dict):
    try:
        dat = t_dict['mcherry']
    except KeyError:
        dat = t_dict['red']
    return dat


def get_measurement_channel(t_dict):
    try:
        dat = t_dict['gcamp']
    except KeyError:
        dat = t_dict['green']
    return dat


def detrend_exponential(y_with_nan):
    """
    Bleach correction via simple exponential fit, subtraction, and re-adding the mean

    Uses np.polyfit on np.log(y), with errors weighted back to the data space. See:

    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    Parameters
    ----------
    y_with_nan

    Returns
    -------

    """

    ind = np.where(~np.isnan(y_with_nan))[0]
    t = np.squeeze(StandardScaler(copy=False).fit_transform(ind.reshape(-1, 1)))
    y = y_with_nan[ind]
    y_log = np.log(y)

    fit_vars = np.polyfit(t, y_log, 1)#, w=np.sqrt(y))

    # Subtract in the original data space
    y_fit = np.exp(fit_vars[0]) * np.exp(t*fit_vars[1])
    y_corrected = y - y_fit + np.mean(y)

    return ind, y_corrected


def detrend_exponential_lmfit(y_with_nan):
    """
    Bleach correction via simple exponential fit, subtraction, and re-adding the mean

    Uses np.polyfit on np.log(y), with errors weighted back to the data space. See:

    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    Parameters
    ----------
    y_with_nan

    Returns
    -------

    """

    mod = ExponentialModel(nan_policy='omit')
    ind = np.where(~np.isnan(y_with_nan))[0]
    x = np.squeeze(StandardScaler(copy=False).fit_transform(ind.reshape(-1, 1)))
    y = y_with_nan[ind]

    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    y_fit = out.eval(x=x)

    y_corrected = y - y_fit + np.nanmean(y)

    return y_corrected, y_fit
