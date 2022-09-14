import numpy as np
import pandas as pd
import sklearn
from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan


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


def correct_trace_using_linear_model(df_red, df_green, neuron_name=None, predictor_names=None):
    # Predict green from time, volume, and red
    if predictor_names is None:
        predictor_names = ["t", "intensity_image", "area", "x", "y"]
    if neuron_name is not None:
        df_green = df_green[neuron_name]
        df_red = df_red[neuron_name]
    green = df_green["intensity_image"]
    # Also add x and y
    if 't' in predictor_names:
        include_t = True
        predictor_names.remove('t')
    else:
        include_t = False
    predictor_vars = [df_red[name] for name in predictor_names]

    num_timepoints = len(green)
    if include_t:
        predictor_vars.append(range(num_timepoints))
    valid_indices = np.logical_not(np.isnan(green))
    # This is important for test videos that are very short
    if valid_indices.value_counts()[True] <= 4:
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
        model = sklearn.linear_model.LinearRegression()
        model.fit(predictor_matrix, green_trace)
        green_predicted = model.predict(predictor_matrix)
        y_result_missing_na = green_trace - green_predicted

        # Align output and input formats
        y_including_na = fill_missing_indices_with_nan(pd.DataFrame(y_result_missing_na),
                                                       expected_max_t=num_timepoints)[0]
        y_result_including_na = pd.Series(list(y_including_na["intensity_image"]))
    return y_result_including_na
