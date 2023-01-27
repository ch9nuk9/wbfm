from enum import IntEnum
import numpy as np


class BehaviorCodes(IntEnum):
    """
    Top-level behaviors that are discretely annotated. Designed to work with Ulises' automatic annotations

    Note that float adds comparison operators
    """
    FWD = -1
    REV = 1
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
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def assert_is_valid(cls, value):
        assert cls.has_value(value), f"Value {value} is not a valid behavioral code ({cls._value2member_map_})"

    @classmethod
    def assert_all_are_valid(cls, vec):
        for v in np.unique(vec):
            cls.assert_is_valid(v)
