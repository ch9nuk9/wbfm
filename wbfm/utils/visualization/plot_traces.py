import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.widgets import TextBox

from wbfm.utils.projects.utils_filenames import get_sequential_filename
from wbfm.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.general.postures.centerline_classes import shade_using_behavior
from matplotlib import transforms
from matplotlib.ticker import NullFormatter
from tqdm.auto import tqdm

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.utils_plot_traces import check_default_names
import matplotlib.style as mplstyle


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
                                 bleach_correct=True,
                                 behavioral_correlation_shading=None,
                                 min_nonnan=None,
                                 share_y_axis=False,
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
        all_modes = ['red', 'green', 'dr_over_r_20', 'ratio', 'linear_model', 'linear_model_experimental']
        opt = dict(project_data=project_data,
                   calculation_mode=calculation_mode,
                   color_using_behavior=color_using_behavior,
                   bleach_correct=bleach_correct)
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        # Also try to remove outliers and filter
        all_modes = ['ratio', 'dr_over_r_20', 'linear_model']
        opt['remove_outliers'] = True
        # for mode in all_modes:
        #     make_grid_plot_using_project(channel_mode=mode, **opt)
        opt['filter_mode'] = 'rolling_mean'
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        # Also do share-y versions, with the filtering
        opt['share_y_axis'] = True
        for mode in all_modes:
            make_grid_plot_using_project(channel_mode=mode, **opt)
        return

    if neuron_names_to_plot is not None:
        neuron_names = neuron_names_to_plot
    else:
        if isinstance(min_nonnan, float):
            neuron_names = project_data.well_tracked_neuron_names(min_nonnan)
        else:
            neuron_names = project_data.neuron_names
    neuron_names.sort()

    # Build functions to make a single subplot
    options = {'channel_mode': channel_mode, 'calculation_mode': calculation_mode, 'filter_mode': filter_mode,
               'remove_outliers': remove_outliers, 'bleach_correct': bleach_correct}
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = lambda axis: project_data.shade_axis_using_behavior(axis)
    logger = project_data.logger

    # Correlate to a behavioral variable
    background_shading_value_func = factory_correlate_trace_to_behavior_variable(project_data,
                                                                                 behavioral_correlation_shading)

    fig = make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func,
                                        color_using_behavior=color_using_behavior,
                                        background_shading_value_func=background_shading_value_func,
                                        logger=logger,
                                        share_y_axis=share_y_axis)

    plt.tight_layout()

    # Save final figure
    if to_save:
        if neuron_names_to_plot is None:
            prefix = f"{channel_mode}_{calculation_mode}"
            if remove_outliers:
                prefix = f"{prefix}_outliers_removed"
            if filter_mode != "no_filtering":
                prefix = f"{prefix}_{filter_mode}"
            if share_y_axis:
                prefix = f"{prefix}_sharey"
            fname = f"{prefix}_grid_plot.png"
        else:
            fname = f"{len(neuron_names_to_plot)}neurons_{channel_mode}_{calculation_mode}_grid_plot.png"
        traces_cfg = project_data.project_config.get_traces_config()
        out_fname = traces_cfg.resolve_relative_path(fname, prepend_subfolder=True)

        save_grid_plot(out_fname)

    return fig


def make_grid_plot_from_dataframe(df: pd.DataFrame,
                                  project_data=None,
                                  neuron_names_to_plot: list = None,
                                  to_save=False,
                                  **kwargs):
    """

    Parameters
    ----------
    df
    project_data - Used for shading from behavioral annotations, if present
    neuron_names_to_plot
    to_save
    kwargs - see make_grid_plot_from_callables

    Returns
    -------

    """

    if neuron_names_to_plot is not None:
        neuron_names = neuron_names_to_plot
    else:
        neuron_names = get_names_from_df(df)
    neuron_names.sort()

    # Build functions to make a single subplot
    tspan = np.arange(df.shape[0])
    get_data_func = lambda neuron_name: (tspan, df[neuron_name])
    if project_data is not None:
        shade_plot_func = lambda axis: project_data.shade_axis_using_behavior(axis)
        logger = project_data.logger
    else:
        shade_plot_func = lambda axis: None
        logger = None

    fig = make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func,
                                        logger=logger,
                                        **kwargs)

    return fig


def make_grid_plot_from_two_dataframes(df0, df1, twinx_when_reusing_figure=True, **kwargs):
    """

    Parameters
    ----------
    df0 - first trace (blue)
    df1 - second trace (orange)
    twinx_when_reusing_figure - Whether to plot the second trace on its own yaxis, or keep the same
    kwargs

    Returns
    -------

    """
    fig = make_grid_plot_from_dataframe(df0, **kwargs)
    fig = make_grid_plot_from_dataframe(df1, fig=fig, twinx_when_reusing_figure=twinx_when_reusing_figure, **kwargs)
    return fig


def factory_correlate_trace_to_behavior_variable(project_data, behavioral_correlation_shading: str) \
        -> Optional[callable]:
    valid_behavioral_shadings = ['absolute_speed', 'speed', 'positive_speed', 'negative_speed', 'curvature']
    posture_class = project_data.worm_posture_class
    y = None
    if behavioral_correlation_shading is None:
        y = None
    elif behavioral_correlation_shading == 'absolute_speed':
        y = posture_class.worm_speed_fluorescence_fps
    elif behavioral_correlation_shading == 'speed':
        y = posture_class.worm_speed_fluorescence_fps_signed
    elif behavioral_correlation_shading == 'positive_speed':
        y = posture_class.worm_speed_fluorescence_fps_signed
        if posture_class.beh_annotation is not None:
            reversal_ind = posture_class.beh_annotation == 1
            y[reversal_ind] = 0
    elif behavioral_correlation_shading == 'negative_speed':
        y = posture_class.worm_speed_fluorescence_fps_signed
        if posture_class.beh_annotation is not None:
            forward_ind = posture_class.beh_annotation == 0
            y[forward_ind] = 0
    elif behavioral_correlation_shading == 'curvature':
        y = posture_class.leifer_curvature_from_kymograph
    else:
        assert behavioral_correlation_shading in valid_behavioral_shadings, \
            f"Must pass None or one of: {valid_behavioral_shadings}"

    if y is None:
        return None

    def background_shading_value_func(X):
        ind = np.where(~np.isnan(X))[0]
        return np.corrcoef(X[ind], y[:len(X)][ind])[0, 1]

    return background_shading_value_func


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

    make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func,
                                  color_using_behavior=color_using_behavior, logger=logger)

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
                                  background_shading_value_func: callable = None,
                                  color_using_behavior: bool = True,
                                  share_y_axis: bool = True,
                                  logger: logging.Logger = None,
                                  num_columns: int = 5,
                                  twinx_when_reusing_figure: bool = False,
                                  fig = None):
    """

    Parameters
    ----------
    color_using_behavior - boolean for actually shading
    get_data_func - function that accepts a neuron name and returns a tuple of (t, y)
    neuron_names - list of neurons to plot
    shade_plot_func - function that accepts an axis object and shades the plot
    background_shading_value_func - function to get a value to shade the background, e.g. correlation to behavior
    color_using_behavior - whether to use the shade_plot_func
    logger

    Example:
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = project_data.shade_axis_using_behavior

    Returns
    -------

    """
    # Set up the colormap of the background, if any
    if background_shading_value_func is not None:
        # From: https://stackoverflow.com/questions/59638155/how-to-set-0-to-white-at-a-uneven-color-ramp

        # First get all the traces, so that the entire cmap can be scaled
        all_y = [get_data_func(name)[1] for name in neuron_names]
        all_vals = [background_shading_value_func(y) for y in all_y]

        norm = TwoSlopeNorm(vmin=np.nanmin(all_vals), vcenter=0, vmax=np.nanmax(all_vals))
        # norm.autoscale(all_vals)
        values_normalized = norm(all_vals)
        colors = plt.cm.PiYG(values_normalized)

    else:
        colors = []

    # Loop through neurons and plot
    num_neurons = len(neuron_names)
    num_rows = int(np.ceil(num_neurons / float(num_columns)))
    if logger is not None:
        logger.info(f"Found {num_neurons} neurons; shaping to grid of shape {(num_rows, num_columns)}")
    if fig is None:
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(25, 25), sharex=True, sharey=share_y_axis)
        new_fig = True
    else:
        new_fig = False

    for i in tqdm(range(len(neuron_names))):
        ax, neuron_name = fig.axes[i], neuron_names[i]
        if twinx_when_reusing_figure and not new_fig:
            ax = ax.twinx().twiny()
            ax_opt = dict(color='tab:orange')
        else:
            ax_opt = dict()
        t, y = get_data_func(neuron_name)

        # print(neuron_name, y.mean())
        if not new_fig:
            ax.plot(t, y, **ax_opt)
        else:
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

            if background_shading_value_func is not None:
                color, val = colors[i], background_shading_value_func(y)
                ax.axhspan(y.min(), y.max(), xmax=len(y), facecolor=color, alpha=0.25, zorder=-100)
                ax.set_title(f"Shaded value: {val:0.2f}")

    return fig


def _plot_subplots(y1, y2):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5), dpi=100)

    ax1 = axes[0]
    ax1.plot(y1, label='Original trace')
    ax1_2 = ax1.twinx()
    ax1_2.plot(y2, 'tab:orange')
    ax1.legend()

    window = [500, 700]
    for w in window:
        ax1.plot([w, w], [np.nanmin(y1), np.nanmax(y1)], color='black', lw=3)

    ax2 = axes[1]
    ax2.plot(y1[window[0]:window[1]], lw=2)
    ax2_2 = ax2.twinx()
    ax2_2.plot(y2[window[0]:window[1]], 'tab:orange', lw=2, label='Modified trace')
    ax2_2.legend()

    return ax1, ax2


def title_from_params(params):
    t = ''
    for key, val in params.items():
        if isinstance(val, str):
            k = key.split('_')[0]
            t += f'{k}={val}-'

    return t[:-1]


def plot_compare_two_calculation_methods(project_data, neuron_name, variable_dict=None, **kwargs):
    """
    kwargs:
        channel_mode: str,
        calculation_mode: str,
        neuron_name: str,
        filter_mode: str = 'no_filtering'
    """

    default_kwargs = dict(
        channel_mode='dr_over_r_20',
        calculation_mode='integration',
        filter_mode='rolling_mean',
        remove_outliers=True
    )
    default_kwargs.update(kwargs)

    t, y1 = project_data.calculate_traces(neuron_name=neuron_name, **default_kwargs)

    for key, val_list in variable_dict.items():
        for val in val_list:
            default_kwargs[key] = val
            t, y2 = project_data.calculate_traces(neuron_name=neuron_name, **default_kwargs)

            ax1, ax2 = _plot_subplots(y1, y2)

            title = title_from_params(default_kwargs)
            ax1.set_title(title)
            ax2.set_title(neuron_name)

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


##
## For interactivity
##

class ClickableGridPlot:
    def __init__(self, project_data, verbose=3):

        # Set up grid plot
        opt = dict(channel_mode='ratio',
                   calculation_mode='integration',
                   filter_mode='rolling_mean',
                   to_save=False)

        mplstyle.use('fast')
        with safe_cd(project_data.project_dir):
            fig = make_grid_plot_using_project(project_data, **opt)

        self.fig = fig
        self.project_data = project_data

        # Set up metadata objects
        names = project_data.neuron_names
        self.selected_neurons = {n: {"List ID": 0, "Proposed Name": n} for n in names}
        self.current_list_index = 1
        self.current_selected_label = None
        self.verbose = verbose


        # Set up text box for modifying names
        # plt.subplots_adjust(bottom=0.2)
        # axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        # self.text_box = TextBox(axbox, 'Modify neuron name', initial="initial_text")
        # self.text_box.on_submit(self.modify_neuron_name)

        # Finish
        self.connect()
        # Load file and add initial colors, if any
        self.load_previous_file()
        plt.show()

    def connect(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.shade_selected_subplot_callback)
        cid = self.fig.canvas.mpl_connect('key_press_event', self.update_current_list_index)
        cid = self.fig.canvas.mpl_connect('close_event', self.write_file)

    def update_current_list_index(self, event):
        if event.key in ['1', '2', '3']:
            self.current_list_index = int(event.key)
        else:
            self.current_list_index = 0

        print(f"Current list index: {self.current_list_index}")

    def modify_neuron_name(self, text):
        self.selected_neurons[self.current_selected_label]["Proposed Name"] = text

    def update_selected_label(self, new_label):
        self.current_selected_label = new_label
        # self.text_box.set_val(new_label)

    def get_color_from_list_index(self):
        print(f"Getting color: {self.current_list_index}")
        if self.current_list_index == 1:
            return 'green'
        elif self.current_list_index == 2:
            return 'blue'
        else:
            return 'red'

    def shade_selected_subplot_callback(self, event):
        ax = event.inaxes
        if self.verbose >= 3:
            print(event)
            print(ax)
        if ax is None or len(ax.lines) == 0:
            return
        button_pressed = event.button

        self.shade_selected_subplot(ax, button_pressed)

    def shade_selected_subplot(self, ax, button_pressed):

        line = ax.lines[0]
        label = line.get_label()
        self.update_selected_label(label)

        # Button codes: https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.MouseButton
        if button_pressed == 1:
            # Left click = select neuron
            if self.selected_neurons[label]["List ID"] == self.current_list_index:
                print(f"{label} already selected")
            else:
                print(f"Selecting {label}")
                self._reset_shading(ax)

                y = line.get_ydata()
                color = self.get_color_from_list_index()

                shading = ax.axhspan(np.nanmin(y), np.nanmax(y), xmax=len(y), facecolor=color, alpha=0.25, zorder=-100)
                ax.draw_artist(shading)

                self.selected_neurons[label]["List ID"] = self.current_list_index

        elif button_pressed == 3:
            # Right click = deselect
            if self.selected_neurons[label]["List ID"] == 0:
                print(f"{label} not selected")
            else:
                print(f"Deselecting {label}")
                self._reset_shading(ax)
                plt.draw()
                self.selected_neurons[label]["List ID"] = 0
        else:
            print("Button press detected, but did nothing")
        # From: https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        ax.figure.canvas.blit(ax.bbox)
        # if verbose >= 2:
        #     print("Currently selected neuron:")
        #     print(self.selected_neurons)

    def _reset_shading(self, ax):
        if len(ax.patches) > 0:
            [p.remove() for p in ax.patches]
            # ax.patches = []

    def write_file(self, event):
        log_dir = self.project_data.project_config.get_visualization_config().absolute_subfolder
        fname = os.path.join(log_dir, 'selected_neurons.csv')
        # fname = get_sequential_filename(fname)
        print(f"Saving: {fname}")

        df = pd.DataFrame(self.selected_neurons)
        df.T.to_csv(path_or_buf=fname, index=True)
        fname = Path(fname).with_suffix('.xlsx')
        df.T.to_excel(fname, index=True)
        # df = pd.DataFrame(self.selected_neurons, index=[0])
        # df.to_csv(path_or_buf=fname, header=True, index=False)

        print(df.T)

    def load_previous_file(self):
        visualization_directory = self.project_data.project_config.get_visualization_config().absolute_subfolder
        fname = os.path.join(visualization_directory, 'selected_neurons.csv')
        if not os.path.exists(fname):
            print(f"Did not find previous state at: {fname}")
            return
        else:
            # plt.show(block=False)
            self.fig.canvas.draw()
            print(f"Reading previous state from: {fname}")
            df = pd.read_csv(fname, index_col=0)

            axes = self.fig.axes
            button_pressed = 1

            for ax, (name, list_index) in zip(axes, df.iterrows()):
                if list_index[0] == 0:
                    continue

                # Add the shading to this axis
                print(f"Shading {name} with index {list_index}")
                self.current_list_index = list_index[0]
                self.shade_selected_subplot(ax, button_pressed)

                # Also add the info to the dict
                self.selected_neurons[name]["List ID"] = list_index

        plt.draw()
        self.current_list_index = 1
