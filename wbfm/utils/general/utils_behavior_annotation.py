from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, make_binary_vector_from_starts_and_ends
from wbfm.utils.general.custom_errors import InvalidBehaviorAnnotationsError


class BehaviorCodes(IntEnum):
    """
    Top-level behaviors that are discretely annotated. Designed to work with Ulises' automatic annotations

    Note that float adds comparison operators
    """
    FWD = -1
    REV = 1
    FWD_VENTRAL_TURN = 2  # Manually annotated
    FWD_DORSAL_TURN = 3  # Manually annotated
    REV_VENTRAL_TURN = 4  # Manually annotated
    REV_DORSAL_TURN = 5  # Manually annotated
    SUPERCOIL = 6  # Manually annotated
    QUIESCENCE = 7  # Manually annotated

    # These don't work properly
    # VENTRAL_TURN = FWD_VENTRAL_TURN | REV_VENTRAL_TURN
    # DORSAL_TURN = FWD_DORSAL_TURN | REV_DORSAL_TURN
    # ALL_TURNS = VENTRAL_TURN | DORSAL_TURN

    NOT_ANNOTATED = 0
    UNKNOWN = -99

    @classmethod
    def shading_cmap(cls):
        """Colormap for shading on top of traces"""
        cmap = {cls.UNKNOWN: None,
                cls.FWD: None,
                cls.REV: 'lightgray'}
        return cmap

    @classmethod
    def base_colormap(cls):
        # See: https://plotly.com/python/discrete-color/
        return px.colors.qualitative.Set1_r

    @classmethod
    def ethogram_cmap(cls, include_reversal_turns=False):
        """Colormap for shading as a stand-alone ethogram"""
        base_cmap = cls.base_colormap()
        cmap = {cls.UNKNOWN: None,
                cls.FWD: base_cmap[0],
                cls.REV: base_cmap[1],
                cls.FWD_VENTRAL_TURN: base_cmap[2],
                cls.FWD_DORSAL_TURN: base_cmap[3],
                # Same as REV
                cls.REV_VENTRAL_TURN: base_cmap[1],
                cls.REV_DORSAL_TURN: base_cmap[1],
                # Unclear
                cls.QUIESCENCE: base_cmap[4],
                }
        if include_reversal_turns:
            cmap[cls.REV_VENTRAL_TURN] = base_cmap[4]
            cmap[cls.REV_DORSAL_TURN] = base_cmap[5]
            cmap[cls.QUIESCENCE] = base_cmap[6]
        return cmap

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def assert_is_valid(cls, value):
        if not cls.has_value(value):
            raise InvalidBehaviorAnnotationsError(f"Value {value} is not a valid behavioral code "
                                                  f"({cls._value2member_map_})")

    @classmethod
    def assert_all_are_valid(cls, vec):
        for v in np.unique(vec):
            cls.assert_is_valid(v)

    @classmethod
    def must_be_manually_annotated(cls, value):
        """As of 23-03-2023, everything except FWD and REV must be manually annotated"""
        if value is None:
            return False
        return value not in (cls.FWD, cls.REV, cls.NOT_ANNOTATED, cls.UNKNOWN)


def options_for_ethogram(beh_vec, shading=False, include_reversal_turns=False):
    """
    Returns a list of dictionaries that can be passed to plotly to draw an ethogram

    if shading is True, then the ethogram will be partially transparent, to be drawn on top of a trace

    Parameters
    ----------
    beh_vec
    shading

    Returns
    -------

    """
    all_shape_opt = []
    if shading:
        cmap = BehaviorCodes.shading_cmap()
    else:
        cmap = BehaviorCodes.ethogram_cmap(include_reversal_turns=include_reversal_turns)

    # Loop over all behaviors in the colormap (some may not be present in the vector)
    for behavior_code, color in cmap.items():
        binary_behavior = beh_vec == behavior_code
        starts, ends = get_contiguous_blocks_from_column(binary_behavior, already_boolean=True)
        for s, e in zip(starts, ends):
            this_val = beh_vec[s]
            color = cmap[this_val]
            # Note that yref is ignored if this is a subplot. If yref is manually set, then it refers to the entire plot
            shape_opt = dict(type="rect", x0=s, x1=e, yref='paper', y0=0, y1=1,
                             fillcolor=color, line_width=0, layer="below")
            all_shape_opt.append(shape_opt)

    return all_shape_opt


def detect_peaks_and_interpolate(dat, to_plot=False, fig=None,
                                 height="mean", width=5):
    """
    Builds a time series approximating the highest peaks of an oscillating signal

    Returns the interpolation class, which has the location and value of the peaks themselves

    Parameters
    ----------
    dat
    to_plot

    Returns
    -------

    """

    # Get peaks
    if height == "mean":
        height = np.mean(dat)
    peaks, properties = find_peaks(dat, height=height, width=width)
    y_peaks = dat[peaks]

    # Interpolate
    interp_obj = interp1d(peaks, y_peaks, kind='cubic', bounds_error=False, fill_value="extrapolate")
    x = np.arange(len(dat))
    y_interp = interp_obj(x)

    if to_plot:
        if fig is None:
            plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(dat, label="Raw data")
        plt.scatter(peaks, y_peaks, c='r', label="Detected peaks")
        plt.plot(x, y_interp, label="Interpolated envelope")#, c='tab:purple')
        plt.title("Envelope signal interpolated between peaks")
        plt.legend()

    return x, y_interp, interp_obj


def approximate_behavioral_annotation_using_pc1(project_cfg):
    """
    Uses the first principal component of the traces to approximate annotations for forward and reversal
    IMPORTANT: Although pc0 should correspond to rev/fwd, the sign of the PC is arbitrary, so we need to check
    that the sign is correct. Currently there's no way to do that without ID'ing a neuron that should correlate to fwd
    or rev, and checking that the sign is correct
    TODO: Add a check for the sign of the PC

    Saves an excel file within the project's behavior folder, and updates the behavioral config

    This file should be found by get_manual_behavior_annotation_fname

    Parameters
    ----------
    project_cfg

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Calculate traces of the project
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=True,
               channel_mode='dr_over_r_50')
    df_traces = project_data.calc_default_traces(**opt)
    from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
    df_traces_no_nan = fill_nan_in_dataframe(df_traces, do_filtering=True)
    # Then PCA
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    pipe.fit(df_traces_no_nan.T)

    # Using a threshold of 0, assign forward and reversal
    pc0 = pipe.steps[1][1].components_[0, :]
    starts, ends = get_contiguous_blocks_from_column(pd.Series(pc0) > 0, already_boolean=True)
    beh_vec = pd.DataFrame(make_binary_vector_from_starts_and_ends(starts, ends, pc0, pad_nan_points=(5, 0)),
                           columns=['Annotation'])
    beh_vec[beh_vec == 1] = BehaviorCodes.REV
    beh_vec[beh_vec == 0] = BehaviorCodes.FWD

    # Save within the behavior folder
    beh_cfg = project_data.project_config.get_behavior_config()
    fname = 'immobilized_beh_annotation'
    beh_cfg.save_data_in_local_project(beh_vec, fname,
                                       prepend_subfolder=True, suffix='.xlsx', sheet_name='behavior')
    beh_cfg.config['manual_behavior_annotation'] = str(Path(fname).with_suffix('.xlsx'))
    beh_cfg.update_self_on_disk()

    return beh_vec
