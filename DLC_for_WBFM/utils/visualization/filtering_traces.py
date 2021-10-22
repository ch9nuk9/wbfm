import logging
import numpy as np
import pandas as pd


def remove_outliers_via_rolling_mean(y: pd.DataFrame, window: int, outlier_threshold=None):
    y_filt = y.rolling(window, min_periods=1, center=True).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 10*error.var() + error.mean()
        print(f"Calculated error threshold at {outlier_threshold}")
        # logging.info(f"Calculated error threshold at {outlier_threshold}")
    is_outlier = error > outlier_threshold
    y[is_outlier] = np.nan

    return y


def remove_outliers_large_diff(y: pd.DataFrame, outlier_threshold=None):
    raise NotImplementedError
    diff = y.diff()
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 10*diff.var() + np.abs(diff.mean())
        print(f"Calculated error threshold at {outlier_threshold}")
    # Only remove the first jump frame, because often the outliers are only one frame
    is_outlier = diff > outlier_threshold
    y[is_outlier] = np.nan

    return y


def filter_rolling_mean(y: pd.DataFrame, window: int = 7):
    return y.rolling(window, min_periods=3).mean()


def filter_linear_interpolation(y: pd.DataFrame, window=15):
    return y.interpolate(method='linear', limit=window, limit_direction='both')
