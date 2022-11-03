import logging
import numpy as np
import pandas as pd

from wbfm.utils.traces.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.visualization.utils_plot_traces import correct_trace_using_linear_model


def remove_outliers_via_rolling_mean(y: pd.Series, window: int, outlier_threshold=None, verbose=0):
    # In practice very sensitive to exact threshold value, which only really works for the ratio
    y = y.copy()
    y_filt = y.rolling(window, min_periods=1, center=True).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 2*error.std() + error.mean()
        if verbose >= 1:
            print(f"Calculated error threshold at {outlier_threshold}")
        # logging.info(f"Calculated error threshold at {outlier_threshold}")
    is_outlier = error > outlier_threshold
    y[is_outlier] = np.nan

    return y


def remove_outliers_using_std(y: pd.Series, std_factor: float, verbose=0):
    y = y.copy()
    outlier_threshold = std_factor * np.std(y)
    is_outlier = np.abs(y - y.mean()) > outlier_threshold
    y[is_outlier] = np.nan
    if verbose >= 1:
        print(f"Removed {len(np.where(is_outlier)[0])} outliers")

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


def filter_rolling_mean(y: pd.DataFrame, window: int = 9):
    return y.rolling(window, min_periods=1).mean()


def filter_linear_interpolation(y: pd.DataFrame, window=15):
    return y.interpolate(method='linear', limit=window, limit_direction='both')


def _get_y_and_vol(df_tmp, i, column_name):
    try:
        y_raw = df_tmp[i][column_name]
        vol = df_tmp[i]['area']
    except KeyError:
        # Then we just have a single level, and don't have the volume
        y_raw = df_tmp[i]
        vol = None
    return y_raw, vol


def _check_valid(y, background_per_pixel):
    if any(y < 0):
        logging.warning(f"Found negative trace value; check background_per_pixel value ({background_per_pixel})")


def trace_from_dataframe_factory(calculation_mode, background_per_pixel, bleach_correct,
                                 preprocess_volume_correction, column_name='intensity_image') -> callable:
    # Way to process a single dataframe
    if calculation_mode == 'integration':
        def calc_single_trace(i, df_tmp) -> pd.Series:

            if preprocess_volume_correction:
                # This function can do everything, including bleach correction
                opt = dict(predictor_names=['t', 'area'], neuron_name=i, remove_intercept=False,
                           bleach_correct=bleach_correct, target_name=column_name)
                y = correct_trace_using_linear_model(df_tmp, **opt)
            else:
                # Otherwise we manually bleach and background correct
                y_raw, vol = _get_y_and_vol(df_tmp, i, column_name)
                if bleach_correct:
                    y_raw = pd.Series(detrend_exponential_lmfit(y_raw, restore_mean_value=True)[0])

                if background_per_pixel > 0:
                    if vol is None:
                        logging.warning("Background subtraction requested, but volume was not included in the dataframe")
                        y = y_raw
                    else:
                        y = y_raw - background_per_pixel * vol
                else:
                    y = y_raw
            _check_valid(y, background_per_pixel)
            return y

    elif calculation_mode == 'mean':
        def calc_single_trace(i, df_tmp) -> pd.Series:
            y_raw, vol = _get_y_and_vol(df_tmp, i, column_name)

            if background_per_pixel > 0:
                if vol is None:
                    logging.warning("Background subtraction requested, but volume was not included in the dataframe")
                    y = y_raw
                else:
                    y = y_raw / vol - background_per_pixel
            else:
                y = y_raw

            _check_valid(y, background_per_pixel)
            return y

    elif calculation_mode == 'volume':
        def calc_single_trace(i, df_tmp) -> pd.Series:
            try:
                y_raw = df_tmp[i]['area']
            except KeyError:
                y_raw = df_tmp[i]['volume']
            return y_raw
    elif calculation_mode == 'z':
        def calc_single_trace(i, df_tmp) -> pd.Series:
            try:
                y_raw = df_tmp[i]['z']
            except KeyError:
                y_raw = df_tmp[i]['z_dlc']
            return y_raw
    elif calculation_mode == 'likelihood':
        def calc_single_trace(i, df_tmp) -> pd.Series:
            y_raw = df_tmp[i]['likelihood']
            return y_raw
    else:
        raise ValueError(f"Unknown calculation mode {calculation_mode}")

    return calc_single_trace
