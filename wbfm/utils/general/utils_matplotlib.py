# From https://stackoverflow.com/questions/48140576/matplotlib-toolbar-in-a-pyqt5-application
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import pearsonr

from wbfm.utils.external.utils_jupyter import executing_in_notebook


def get_twin_axis(ax, axis='x'):
    # From: https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
    assert axis in ("x", "y")
    siblings = getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None


def paired_boxplot_from_dataframes(df_before_and_after: pd.DataFrame, labels: list = None,
                                   num_rows = 2,
                                   add_median_line=True,
                                   use_coloring=True,
                                   fig=None):
    """
    Plots a pair of boxplots with red (green) lines showing points that lost (gained) value between conditions

    Note that conditions must be paired for this to work

    Parameters
    ----------
    df_before_and_after: num_rows x n Dataframe. Index is the x position on the boxplot (i.e. long form data)
        Note that more rows can be present, but they will be ignored
        Rows are compared based on position (0th row is before, 1st row is after)
    labels

    Returns
    -------

    """
    box_opt = {}
    if labels is not None:
        box_opt['labels'] = labels
    if fig is None:
        fig = plt.figure(dpi=100)
    x = df_before_and_after.index[:num_rows]
    y_vec_list = []
    for i in range(num_rows):
        y_vec_list.append(df_before_and_after.iloc[i, :])
    positions = np.arange(num_rows)
    bplot = plt.boxplot(y_vec_list, positions=positions, zorder=10, patch_artist=True, **box_opt)
    plt.xticks(ticks=positions, labels=x)

    # Plot lines between each pair of boxes
    for i in range(num_rows - 1):
        y0_vec = y_vec_list[i]
        y1_vec = y_vec_list[i + 1]
        this_x = x[i:i+2]
        diff = y1_vec - y0_vec
        if use_coloring:
            colors = ['green' if d > 0 else 'red' for d in diff]
        else:
            colors = ['black' for _ in diff]
        for y0, y1, col in zip(y0_vec, y1_vec, colors):
            plt.plot(this_x, [y0, y1], color=col, alpha=0.1)
        for patch in bplot['boxes']:
            patch.set_facecolor('lightgray')

        if add_median_line:
            y0_med, y1_med = np.median(y0_vec), np.median(y1_vec)
            col = 'green' if y1_med > y0_med else 'red'
            plt.plot([i, i+1], [y0_med, y1_med], color=col, zorder=20, linewidth=2)

    return fig


def corrfunc(x, y, ax=None, **kws):
    """
    Plot the correlation coefficient in the top right hand corner of a plot.
    If there are multiple colors, offsets the text below

    Can be used with seaborn pairplots using map_lower(corrfunc)

    From: https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot-using-seaborn-in-python

    Parameters
    ----------
    x
    y
    ax
    kws

    Returns
    -------

    """
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    offset = 0
    for c in ax.get_children():
        if isinstance(c, matplotlib.text.Annotation):
            offset += 0.075
    ax.annotate(f'ρ={r:.2f}', xy=(.8, .9 - offset), xycoords=ax.transAxes)


def build_histogram_from_counts(all_dat, pixel_sz=0.1):
    """
    From 3d count data (e.g. x and y over time), convert into a video

    Parameters
    ----------
    all_dat - format: txy

    Returns
    -------

    """

    bins = int(np.ceil(2.0 / pixel_sz))
    video_histogram = np.zeros((len(all_dat), bins, bins))

    for i, dat in enumerate(all_dat):
        video_histogram[i, ...] = np.histogram2d(dat[:, 0], dat[:, 1], bins=bins)[0]

    return video_histogram


def check_plotly_rendering(X: np.ndarray, max_size_for_notebook=200) -> (bool, dict):
    if not executing_in_notebook():
        return False
    if any(np.array(X.shape) > max_size_for_notebook):
        static_rendering_required = True
        logging.warning(
            f"Plotly will crash jupyter notebook if > {max_size_for_notebook} neurons (there are {X.shape}). "
            "Will render static image instead.")
    else:
        static_rendering_required = False

    if static_rendering_required:
        render_opt = dict(renderer="svg")
    else:
        render_opt = dict()
    return static_rendering_required, render_opt
