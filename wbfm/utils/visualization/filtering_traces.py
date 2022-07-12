import logging
import numpy as np
import pandas as pd
from wbfm.utils.visualization.napari_from_config import napari_labels_from_frames
from wbfm.utils.visualization.napari_utils import napari_labels_from_traces_dataframe
from wbfm.utils.general.postures.centerline_classes import shade_using_behavior
from scipy.spatial.distance import cdist


def remove_outliers_via_rolling_mean(y: pd.DataFrame, window: int, outlier_threshold=None):
    # In practice very sensitive to exact threshold value, which only really works for the ratio
    y_filt = y.rolling(window, min_periods=1, center=True).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 5*error.std() + error.mean()
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


def trace_from_dataframe_factory(calculation_mode, background_per_pixel):
    # Way to process a single dataframe
    if calculation_mode == 'integration':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['intensity_image']
                vol = df_tmp[i]['area']
            except KeyError:
                y_raw = df_tmp[i]['brightness']
                vol = df_tmp[i]['volume']
            y = y_raw - background_per_pixel * vol
            if any(y < 0):
                logging.warning(f"Found negative trace value; check background_per_pixel value ({background_per_pixel})")
            return y
    # elif calculation_mode == 'max':
    #     def calc_single_trace(i, df_tmp):
    #         y_raw = df_tmp[i]['all_values']
    #         f = lambda x: np.max(x, initial=np.nan)
    #         return y_raw.apply(f) - background_per_pixel
    elif calculation_mode == 'mean':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['intensity_image']
                vol = df_tmp[i]['area']
            except KeyError:
                y_raw = df_tmp[i]['brightness']
                vol = df_tmp[i]['volume']
            y = y_raw / vol - background_per_pixel
            if any(y < 0):
                logging.warning(f"Found negative trace value; check background_per_pixel value ({background_per_pixel})")
            return y
    # elif calculation_mode == 'quantile90':
    #     def calc_single_trace(i, df_tmp):
    #         y_raw = df_tmp[i]['all_values']
    #         return np.quantile(y_raw, 0.9) - self.background_per_pixel
    # elif calculation_mode == 'quantile50':
    #     def calc_single_trace(i, df_tmp):
    #         y_raw = df_tmp[i]['all_values']
    #         f = lambda x: np.quantile(x, initial=np.nan)
    #         return np.quantile(y_raw, 0.5) - self.background_per_pixel
    elif calculation_mode == 'volume':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['area']
            except KeyError:
                y_raw = df_tmp[i]['volume']
            return y_raw
    elif calculation_mode == 'z':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['z']
            except KeyError:
                y_raw = df_tmp[i]['z_dlc']
            return y_raw
    elif calculation_mode == 'likelihood':
        def calc_single_trace(i, df_tmp):
            y_raw = df_tmp[i]['likelihood']
            return y_raw
    else:
        raise ValueError(f"Unknown calculation mode {calculation_mode}")

    return calc_single_trace
