from enum import IntEnum


class BehaviorCodes(IntEnum):
    """Top-level behaviors that are discretely annotated. Designed to work with Ulises' automatic annotations"""
    FWD = -1
    REV = 1
    UNKNOWN = 0

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
