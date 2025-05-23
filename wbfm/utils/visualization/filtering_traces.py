import logging
import numpy as np
import pandas as pd

from wbfm.utils.external.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.visualization.utils_plot_traces import correct_trace_using_linear_model


def remove_outliers_via_rolling_mean(y: pd.Series, window: int, outlier_threshold=None, std_factor=2, fill_value=np.nan, verbose=0):

    """
    Remove outliers using the rolling mean

    Specifically, remove all values that are more than outlier_threshold away from the rolling mean
    The default outlier_threshold is 2*std + mean

    Parameters
    ----------
    y
    window
    outlier_threshold
    verbose

    Returns
    -------

    """
    
    # In practice very sensitive to exact threshold value, which only really works for the ratio
    y = y.copy()
    y_filt = y.rolling(window, min_periods=1, center=True).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # logging.warning("not working very well")
        outlier_threshold = std_factor*error.std() + error.mean()
        if verbose >= 1:
            print(f"Calculated error threshold at {outlier_threshold}")
        # logging.info(f"Calculated error threshold at {outlier_threshold}")
    is_outlier = error > outlier_threshold
    y[is_outlier] = fill_value

    return y


def remove_outliers_using_std(y: pd.Series, std_factor: float, verbose=0, fill_value='nan'):
    """
    Remove outliers using the standard deviation of the trace

    Specifically, remove all values that are more than std_factor*std away from the mean (constant value)

    Parameters
    ----------
    y
    std_factor
    verbose
    fill_value

    Returns
    -------

    """
    if isinstance(fill_value, str):
        if fill_value == 'nan':
            fill_value = np.nan
        elif fill_value == 'mean':
            fill_value = np.mean(y)
        else:
            raise NotImplementedError
    y = y.copy()
    outlier_threshold = std_factor * np.std(y)
    is_outlier = np.abs(y - y.mean()) > outlier_threshold
    y[is_outlier] = fill_value
    if verbose >= 1:
        print(f"Removed {len(np.where(is_outlier)[0])} outliers")

    return y


def remove_outliers_large_diff(y: pd.DataFrame, outlier_threshold=None):
    raise NotImplementedError
    logging.warning("not working very well")
    diff = y.diff()
    if outlier_threshold is None:
        outlier_threshold = 10*diff.var() + np.abs(diff.mean())
        print(f"Calculated error threshold at {outlier_threshold}")
    # Only remove the first jump frame, because often the outliers are only one frame
    is_outlier = diff > outlier_threshold
    y[is_outlier] = np.nan

    return y


def filter_rolling_mean(y: pd.Series, window: int = 9) -> pd.Series:
    return y.rolling(window, min_periods=1, center=True).mean()


def filter_gaussian_moving_average(y: pd.Series, std=1, window=100) -> pd.Series:
    return y.rolling(center=True, window=window, win_type='gaussian', min_periods=1).mean(std=std)


def filter_exponential_moving_average(y: pd.Series, span=17) -> pd.Series:
    return y.ewm(span=span, min_periods=1).mean()


def filter_tv_diff(y: pd.Series, gamma=0.0015):
    """Gamma chosen by manual inspection of sharp and shallow traces"""
    import pynumdiff # Import locally because it has a loud warning about cvxpy
    dt = 0.1
    iterations = 1
    params = [iterations, gamma]
    x_hat, dxdt_hat = pynumdiff.total_variation_regularization.iterative_velocity(y, dt, params)
    return x_hat


def filter_linear_interpolation(y: pd.Series, window=15):
    return y.interpolate(method='linear', limit=window, limit_direction='both')


def _get_y_and_vol(df_tmp, i, column_name):
    try:
        _df = df_tmp[i]
        y_raw = _df[column_name]
        if 'area' in _df:
            vol = _df['area']
        else:
            vol = None
    except KeyError:
        # Then we just have a single level, and don't have the volume
        y_raw = df_tmp[i]
        vol = None
    return y_raw, vol


def _check_valid(y, background_per_pixel):
    if any(y < 0):
        logging.debug(f"Found negative trace value; check background_per_pixel value ({background_per_pixel})")


def trace_from_dataframe_factory(calculation_mode, background_per_pixel, bleach_correct,
                                 preprocess_volume_correction, column_name='intensity_image') -> callable:
    """
    Builds a function with the following signature:
        calc_single_trace(column_name, df) -> pd.Series

    In other words, takes a dataframe of metadata regarding one trace, and preprocesses the activity time series using,
    for example, linear regression

    Alternatively, can simply return one of those metadata variables

    The default is calculation_mode='integration', which does the following:
    1. Subtract background_per_pixel*vol
    2. Divide the trace by a fitted exponential

    Parameters
    ----------
    calculation_mode
    background_per_pixel: More stable version of preprocess_volume_correction. Background value is measured externally
    bleach_correct: Divide by an exponential fit. Generally True
    preprocess_volume_correction: Experimental, generally False
    column_name

    Returns
    -------

    """
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

                if background_per_pixel is not None and background_per_pixel > 0:
                    if vol is None:
                        logging.warning("Background subtraction requested, but volume was not included in the dataframe")
                        y = y_raw
                    else:
                        y = y_raw - background_per_pixel * vol
                else:
                    y = y_raw

                if bleach_correct:
                    y = pd.Series(detrend_exponential_lmfit(y, restore_mean_value=True)[0])

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


def fill_nan_in_dataframe(df, do_filtering=True):
    if do_filtering:
        df = filter_rolling_mean(df.copy(), window=3)
    df = df.copy().interpolate()
    df.fillna(df.mean(), inplace=True)  # Should only be edges
    return df


def fast_slow_decomposition(y, fast_window=0, slow_window=9):
    """
    Decompose the trace into fast and slow components

    Parameters
    ----------
    y
    fast_window: std of gaussian filter. If 0, then no filtering on the fast component
    slow_window

    Returns
    -------

    """
    assert fast_window < slow_window, "Fast window must be smaller than slow window"
    if fast_window > 0:
        y_fast = filter_gaussian_moving_average(y, std=fast_window)
    else:
        y_fast = y
    y_slow = filter_gaussian_moving_average(y, std=slow_window)
    y_fast = y_fast - y_slow
    return y_fast, y_slow


def filter_trace_using_mode(y, filter_mode="no_filtering"):
    if filter_mode == "rolling_mean":
        y = filter_rolling_mean(y, window=3)
    elif filter_mode == "strong_rolling_mean":
        y = filter_rolling_mean(y, window=5)
    elif filter_mode == "gaussian_moving_average":
        y = filter_gaussian_moving_average(y, std=1, window=5)  # Large window will fill too many nan values
    elif filter_mode == "linear_interpolation":
        y = filter_linear_interpolation(y, window=15)
    elif filter_mode == "3d_pca":
        y = filter_exponential_moving_average(y)
    elif filter_mode == "tvdiff":
        assert all(~np.isnan(y)), "tvdiff doesn't work with nans"
        y = filter_tv_diff(y)
    elif filter_mode == "no_filtering" or filter_mode is None or filter_mode == "":
        pass
    else:
        logging.warning(f"Unrecognized filter mode: {filter_mode}")
    return y
