import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from DLC_for_WBFM.utils.external.utils_pandas import cast_int_or_nan
from DLC_for_WBFM.utils.general.postures.centerline_classes import shade_using_behavior
from matplotlib import transforms
from matplotlib.ticker import NullFormatter
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.visualization.utils_plot_traces import build_trace_factory, check_default_names, set_big_font


##
## New functions for use with project_config files
##

def make_grid_plot_using_project(project_data: ProjectData,
                                 channel_mode: str,
                                 calculation_mode: str,
                                 neuron_names_to_plot: list = None,
                                 filter_mode: str = 'no_filtering',
                                 color_using_behavior=True,
                                 to_save=True):
    if channel_mode == 'all':
        all_modes = ['red', 'green', 'ratio']
        opt = dict(project_data=project_data,
                   calculation_mode=calculation_mode,
                   color_using_behavior=color_using_behavior)
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
    options = {'channel_mode': channel_mode, 'calculation_mode': calculation_mode, 'filter_mode': filter_mode}
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = lambda axis: project_data.shade_axis_using_behavior(axis)
    logger = project_data.logger

    make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func, color_using_behavior, logger)

    # Save final figure
    if to_save:
        if neuron_names_to_plot is None:
            fname = f"{channel_mode}_{calculation_mode}_grid_plot.png"
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


def OLD_make_grid_plot_from_project(traces_config,
                                    trace_mode=None, do_df_over_f0=False, smoothing_func=None,
                                    color_using_behavior=True,
                                    background_per_pixel=15):
    """
    Should be run within a project folder
    """

    assert (trace_mode in ['green', 'red', 'ratio']), f"Unknown trace mode {trace_mode}"

    base_trace_fname = Path(traces_config['traces']['red'])

    # Read in the data
    if smoothing_func is None:
        smoothing_func = lambda x: x
        smoothing_str = ""
    else:
        smoothing_str = "smoothing"

    get_y_raw, neuron_names = build_trace_factory(base_trace_fname, trace_mode, smoothing_func, background_per_pixel)

    # Define df / f0 postprocessing (normalizing) step
    if do_df_over_f0:
        def get_y(i):
            y_raw = get_y_raw(i)
            return y_raw / np.nanquantile(y_raw, 0.1)
    else:
        get_y = get_y_raw

    # Guess a good shape for subplots
    neuron_names.sort()

    num_neurons = len(neuron_names)
    num_columns = 4
    num_rows = num_neurons // num_columns + 1
    print(f"Found {num_neurons} neurons; shaping to grid of shape {(num_rows, num_columns)}")

    # Get axes
    # xlim = [0, max([len(df[i]) for i in neuron_names])]
    # ylim = [0, max([max(df[i]['brightness']/df[i]['volume']) for i in neuron_names])]

    # Loop through neurons and plot
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(45, 15), sharex=True, sharey=False)

    for ax, i_neuron in tqdm(zip(fig.axes, neuron_names)):
        y = get_y(i_neuron)
        ax.plot(y, label=i_neuron)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        ax.set_title(i_neuron, {'fontsize': 28}, y=0.7)
        # ax.legend()
        ax.set_frame_on(False)
        ax.set_axis_off()

    # Save final figure
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.0,
                        hspace=0.0)

    out_fname = base_trace_fname.with_name(f"{smoothing_str}_{trace_mode}_grid_plot.png")
    plt.savefig(out_fname, bbox_inches='tight', pad_inches=0)


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


def visualize_mcherry_and_gcamp(t_dict,
                                name,
                                which_neuron,
                                make_new_fig=True,
                                make_new_title=True,
                                ax1=None,
                                ax2=None,
                                to_normalize=False,
                                preprocess_func=None):
    """
    NOTE: preprocess_func is nonfunctional
    """
    if make_new_fig:
        plt.figure(figsize=(35, 5))  # , fontsize=12)

    if make_new_fig:
        ax1 = plt.subplot(121)
    dat = get_tracking_channel(t_dict)
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax1.plot(dat)
        plt.title(f'Red channel for neuron {name}')
    else:
        ax1.plot(dat, label=f'Red channel for neuron {name}')
        ax1.legend()

    if make_new_fig:
        ax2 = plt.subplot(122)
    dat = get_measurement_channel(t_dict)
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax2.plot(dat)
        plt.title(f'Green channel for neuron {name}')
    else:
        ax2.plot(dat, label=f'Green channel for neuron {name}')
        ax2.legend()

    set_big_font()

    return ax1, ax2


def visualize_ratio(t_dict,
                    name,
                    which_neuron,
                    tspan=None,
                    background=[0, 0],
                    ylim=[0, 1],
                    preprocess_func=None):
    """
    Divides the green by the red channel to produce a normalized time series
        Optionally subtracts a background value
    """
    plt.figure(figsize=(35, 5))

    red = get_tracking_channel(t_dict)
    green = get_measurement_channel(t_dict)

    if preprocess_func is not None:
        red = preprocess_func(red, which_neuron)
        green = preprocess_func(green, which_neuron)

    dat = (green - background[0]) / (red - background[1])
    if tspan is None:
        plt.plot(dat)
    else:
        plt.plot(tspan, dat)
    plt.xlabel('Seconds')
    plt.ylim(ylim)
    plt.title(f"Ratiometric for neuron {name}")


def visualize_all_traces(all_traces,
                         all_names=None,
                         plot_subfunction=visualize_mcherry_and_gcamp,
                         preprocess_func=None,
                         to_save=False):
    """
    Plots all neurons in a struct using a subfunction with the following API:
        plot_subfunction(t_dict,
                         which_neuron=i,
                         name=all_names[i],
                         preprocess_func=preprocess_func)
    """
    all_names = check_default_names(all_names, len(all_traces))

    for i, t_dict in enumerate(all_traces):
        plot_subfunction(t_dict,
                         which_neuron=i,
                         name=all_names[i],
                         preprocess_func=preprocess_func)
        if to_save:
            plt.savefig(f'traces_{all_names[i]}')


##
## Generally plotting
##


# def plot2d_with_max(dat, t, max_ind, max_vals, vmin=100, vmax=400):
#     plt.imshow(dat[:,:,0,t], vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     x, y = max_ind[t,1], max_ind[t,0]
#     if z == max_ind[t,2]:
#         plt.scatter(x, y, marker='x', c='r')
#     plt.title(f"Max for t={t} is {max_vals[t]} xy={x},{y}")

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

# def nan_tracking_failures(config,
#                           dat,
#                           which_neuron,
#                           threshold=0.9):
#     c = load_config(config)
#
#     _, this_prob = xy_from_dlc_dat(c.tracking.annotation_fname,
#                                    which_neuron,
#                                    c.preprocessing.num_frames)
#
#     bad_vals = np.array(this_prob) < threshold
#     dat[bad_vals] = np.nan
#     # print(np.count_nonzero(this_prob < threshold))
#
#     return dat
