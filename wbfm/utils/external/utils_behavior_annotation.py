from enum import IntEnum
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column
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
        return px.colors.qualitative.Bold_r

    @classmethod
    def ethogram_cmap(cls):
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
                cls.QUIESCENCE: base_cmap[4]}
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
        return value not in (cls.FWD, cls.REV, cls.NOT_ANNOTATED, cls.UNKNOWN)


def options_for_ethogram(beh_vec, shading=False):
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
        cmap = BehaviorCodes.ethogram_cmap()

    # Loop over all behaviors in the colormap (some may not be present in the vector)
    for behavior_code, color in cmap.items():
        binary_behavior = beh_vec == behavior_code
        starts, ends = get_contiguous_blocks_from_column(binary_behavior, already_boolean=True)
        for s, e in zip(starts, ends):
            this_val = beh_vec[s]
            color = cmap[this_val]
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
