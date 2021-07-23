import pandas as pd


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

    font = {'weight' : 'bold',
            'size'   : size}
    matplotlib.rc('font', **font)