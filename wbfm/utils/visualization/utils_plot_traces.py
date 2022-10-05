import numpy as np
import pandas as pd
import sklearn
from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.traces.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


def build_trace_factory(base_trace_fname, trace_mode, smoothing_func=lambda x: x, background_per_pixel=0):
    if trace_mode in ['red', 'green']:
        fname = base_trace_fname.with_name(f"{trace_mode}_traces.h5")
        df = pd.read_hdf(fname)
        neuron_names = list(set(df.columns.get_level_values(0)))

        def get_y_raw(i):
            y_raw = df[i]['brightness']
            return smoothing_func(y_raw - background_per_pixel * df[i]['volume'])

    else:
        fname = base_trace_fname.with_name("red_traces.h5")
        df_red = pd.read_hdf(fname)
        fname = base_trace_fname.with_name("green_traces.h5")
        df_green = pd.read_hdf(fname)
        neuron_names = list(set(df_green.columns.get_level_values(0)))

        def get_y_raw(i):
            red_raw = df_red[i]['brightness']
            green_raw = df_green[i]['brightness']
            vol = df_green[i]['volume']  # Same for both
            return smoothing_func((green_raw - vol * background_per_pixel) / (red_raw - vol * background_per_pixel))
    print(f"Read traces from: {fname}")

    return get_y_raw, neuron_names


def check_default_names(all_names, num_neurons):
    if all_names is None:
        all_names = [str(i) for i in range(num_neurons)]
    return all_names


def set_big_font(size=22):
    # From: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    import matplotlib

    font = {'weight': 'bold',
            'size': size}
    matplotlib.rc('font', **font)


def correct_trace_using_linear_model(df_red, df_green, neuron_name=None, predictor_names=None,
                                     remove_intercept=True,
                                     model=sklearn.linear_model.LinearRegression(),
                                     bleach_correct=False,
                                     DEBUG=False):
    """
    Predict green from time, volume, and red

    Note: the indices should start from 0

    Can also use special (calculated) predictors. Currently implemented are:
        t - a simple time vector
        "{var1}_over_{var2}" - dividing the columns given by var1 and var2
        "{var1}_squared" - the square of a variable in df_red. t also works

    Parameters
    ----------
    df_red
    df_green
    neuron_name - Optional. If not passed, assumes the dataframe is not multiindexed
    predictor_names - list of column names to extract from df_red
    remove_intercept - whether to remove the intercept of the linear model
    model

    Returns
    -------

    """
    if predictor_names is None:
        predictor_names = ["t", "intensity_image", "area", "x", "y"]
    if neuron_name is not None:
        df_green = df_green[neuron_name]
        df_red = df_red[neuron_name]
    green = df_green["intensity_image"]
    if bleach_correct:
        green = detrend_exponential_lmfit(green)[0]
    # Construct processed predictors
    processed_vars = []
    simple_predictor_names = []
    for name in predictor_names:
        if '_over_' in name:
            var1, var2 = name.split('_over_')
            this_var = df_red[var1] / df_red[var2]
            processed_vars.append(this_var)
        elif 'intensity_image' in name and bleach_correct:
            red = detrend_exponential_lmfit(df_red[name])[0]
            processed_vars.append(red)
        elif name == 't':
            processed_vars.append(np.arange(len(green)))
        elif '_squared' in name:
            pow = 2.0
            sub_name = name.split('_squared')[0]
            if sub_name in df_red:
                var = df_red[sub_name] ** pow
            elif 't' in sub_name:
                var = np.arange(len(green)) ** pow
            else:
                raise NotImplementedError
            processed_vars.append(var)
        else:
            simple_predictor_names.append(name)

    # Build simple predictors and combine with processed
    predictor_vars = [df_red[name] for name in simple_predictor_names]
    predictor_vars.extend(processed_vars)

    # Get valid indices in all variables
    to_remove_predictors = [np.where(np.isnan(var))[0] for var in predictor_vars]
    to_remove_y = np.where(np.isnan(green))[0]
    to_remove_predictors.append(to_remove_y)
    to_keep = set(range(len(green)))

    to_remove_all = set.union(*[set(r) for r in to_remove_predictors])
    valid_indices = np.array(list(to_keep - to_remove_all))

    # Fix nan values and fit
    # if valid_indices.value_counts()[True] <= 4:
    if len(valid_indices) <= 4:
        # This is important for test videos that are very short
        y_result_including_na = green.copy()
        y_result_including_na[:] = np.nan
    else:

        # remove nas and z score
        def _z_score(_x):
            _x = np.array(_x)[valid_indices]
            return (_x - np.mean(_x)) / np.std(_x)

        green_trace = green[valid_indices]
        predictor_matrix = np.array([_z_score(var) for var in predictor_vars])
        predictor_matrix = np.c_[predictor_matrix.T]

        # create model
        model.fit(predictor_matrix, green_trace)
        if not remove_intercept:
            model.intercept_ = [0.0]
        green_predicted = model.predict(predictor_matrix)
        y_result_missing_na = green_trace - green_predicted

        # Align output and input formats
        # y_df_missing_na = pd.DataFrame(y_result_missing_na, index=np.where(valid_indices)[0])
        y_df_missing_na = pd.DataFrame(y_result_missing_na, index=valid_indices)
        y_including_na = fill_missing_indices_with_nan(y_df_missing_na,
                                                       expected_max_t=len(green))[0]
        # try:
        col_name = get_names_from_df(y_including_na)[0]
        y_result_including_na = pd.Series(list(y_including_na[col_name]))
        # except KeyError:
        #     y_result_including_na = y_including_na
    return y_result_including_na


def get_lower_bound_values(x, y, min_vals_per_bin=10, num_bins=100):
    """
    Calculates the lower bound of a scatter plot

    Designed to be used when high values correspond to anomalies and/or "real" activity that shouldn't be regressed out

    Parameters
    ----------
    x
    y
    min_vals_per_bin
    num_bins

    Returns
    -------

    """
    window_starts = np.linspace(np.min(x), np.max(x), num_bins)
    y_mins = []
    x_mins = window_starts[:-1]
    for i in range(len(window_starts[:-1])):
        start, end = window_starts[i], window_starts[i + 1]
        _ind = np.where(np.logical_and(x < end, x > start))[0]
        if len(_ind) > min_vals_per_bin:
            y_mins.append(np.min(y[_ind]))
        else:
            y_mins.append(0)

    y_mins = np.array(y_mins)
    nonzero_ind = y_mins > 0
    y_mins = y_mins[nonzero_ind]
    x_mins = x_mins[nonzero_ind]

    return x_mins, y_mins
