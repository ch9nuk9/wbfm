# From https://stackoverflow.com/questions/48140576/matplotlib-toolbar-in-a-pyqt5-application

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import pearsonr


def get_twin_axis(ax, axis='x'):
    # From: https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
    assert axis in ("x", "y")
    siblings = getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None


def paired_boxplot_from_dataframes(both_maxes: pd.DataFrame, labels: list=None):
    """
    Plots a pair of boxplots with red (green) lines showing points that lost (gained) value between conditions

    Parameters
    ----------
    both_maxes - 2 x n Dataframe, with index equal to x positions
    labels

    Returns
    -------

    """
    box_opt = {}
    if labels is not None:
        box_opt['labels'] = labels
    plt.figure(dpi=100)
    x = both_maxes.index
    y0_vec = both_maxes.iloc[0, :]
    y1_vec = both_maxes.iloc[1, :]
    diff = y1_vec - y0_vec
    colors = ['green' if d > 0 else 'red' for d in diff]
    for y0, y1, col in zip(y0_vec, y1_vec, colors):
        plt.plot(x, [y0, y1], color=col, alpha=0.5)
    bplot = plt.boxplot([y0_vec, y1_vec], positions=x, zorder=10, patch_artist=True, **box_opt)
    for patch in bplot['boxes']:
        patch.set_facecolor('lightgray')


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
