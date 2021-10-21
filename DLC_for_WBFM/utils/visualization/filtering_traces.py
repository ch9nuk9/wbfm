import logging
import numpy as np
import pandas as pd


def remove_outliers_via_rolling_mean(y: pd.DataFrame, window: int, outlier_threshold=None):
    y_filt = y.rolling(window, min_periods=1).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 100*error.var()
        print(f"Calculated error threshold at {outlier_threshold}")
        # logging.info(f"Calculated error threshold at {outlier_threshold}")
    is_outlier = error > outlier_threshold
    y[is_outlier] = np.nan

    return y


def filter_rolling_mean(y: pd.DataFrame, window: int = 7):
    return y.rolling(window, min_periods=3).mean()


def filter_linear_interpolation(y: pd.DataFrame, window=15):
    return y.interpolate(method='linear', limit=window, limit_direction='both')
