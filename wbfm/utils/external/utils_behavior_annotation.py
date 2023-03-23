from enum import IntEnum
import numpy as np
import plotly.express as px

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column


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
    NOT_ANNOTATED = 0
    UNKNOWN = -99
    # UNKNOWN = NOT_ANNOTATED | GAP

    @classmethod
    def cmap(cls):
        """Colormap for shading on top of traces"""
        cmap = {cls.UNKNOWN: None,
                cls.FWD: None,
                cls.REV: 'lightgray'}
        return cmap

    @classmethod
    def ethogram_cmap(cls):
        """Colormap for shading as a stand-alone ethogram"""
        base_cmap = px.colors.qualitative.Plotly
        cmap = {cls.UNKNOWN: None,
                cls.FWD: base_cmap[0],
                cls.REV: base_cmap[1]}
        return cmap

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def assert_is_valid(cls, value):
        assert cls.has_value(value), f"Value {value} is not a valid behavioral code ({cls._value2member_map_})"

    @classmethod
    def assert_all_are_valid(cls, vec):
        for v in np.unique(vec):
            cls.assert_is_valid(v)


def options_for_ethogram(beh_vec):
    all_shape_opt = []
    cmap = BehaviorCodes.ethogram_cmap()

    # Reversals
    bin_behavior = beh_vec == BehaviorCodes.REV
    starts, ends = get_contiguous_blocks_from_column(bin_behavior, already_boolean=True)
    for s, e in zip(starts, ends):
        this_val = beh_vec[s]
        color = cmap[this_val]
        shape_opt = dict(type="rect", x0=s, x1=e, y0=0, y1=1, fillcolor=color,
                         line_width=0)
        all_shape_opt.append(shape_opt)

    # Forwards
    bin_behavior = beh_vec == BehaviorCodes.FWD
    starts, ends = get_contiguous_blocks_from_column(bin_behavior, already_boolean=True)
    for s, e in zip(starts, ends):
        this_val = beh_vec[s]
        color = cmap[this_val]
        shape_opt = dict(type="rect", x0=s, x1=e, y0=0, y1=1, fillcolor=color,
                         line_width=0)
        all_shape_opt.append(shape_opt)

    return all_shape_opt
