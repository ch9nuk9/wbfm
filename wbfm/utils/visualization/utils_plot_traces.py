import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.linear_model import LinearRegression

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan, get_contiguous_blocks_from_column
from wbfm.utils.general.utils_paper import paper_trace_settings, apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.external.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.external.utils_plotly import hex2rgba, float2rgba
import plotly.graph_objects as go
import plotly.express as px


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


def correct_trace_using_linear_model(df_red: pd.DataFrame,
                                     df_green: Union[pd.DataFrame, np.ndarray, pd.Series]=None,
                                     neuron_name: Optional[str]=None,
                                     predictor_names: Optional[list]=None,
                                     target_name='intensity_image',
                                     remove_intercept=True,
                                     model=LinearRegression(),
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
    if df_green is None:
        df_green = df_red
    if neuron_name is not None:
        if neuron_name in df_green:
            df_green = df_green[neuron_name]
        df_red = df_red[neuron_name]
    if target_name in df_green:
        green = df_green[target_name]
    else:
        green = df_green
    if bleach_correct:
        green = detrend_exponential_lmfit(green, restore_mean_value=True)[0]
    # Construct processed predictors
    processed_vars = []
    simple_predictor_names = []
    for name in predictor_names:
        if '_over_' in name:
            # Division; doesn't work with squared
            var1, var2 = name.split('_over_')
            this_var = df_red[var1] / df_red[var2]
            processed_vars.append(this_var)
        elif '_times_' in name:
            # Cross terms; doesn't work with squared
            var1, var2 = name.split('_times_')
            this_var = df_red[var1] * df_red[var2]
            processed_vars.append(this_var)
        elif name == 'intensity_image' and bleach_correct:
            red = detrend_exponential_lmfit(df_red[name])[0]
            processed_vars.append(red)
        elif name == 't':
            processed_vars.append(np.arange(len(green)))
        elif '_squared' in name or '_cubed' in name:
            # Only power 2 and 3 implemented
            if '_squared' in name:
                pow = 2.0
                sub_name = name.split('_squared')[0]
            else:
                pow = 3.0
                sub_name = name.split('_cubed')[0]

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


def modify_dataframe_to_allow_gaps_for_plotly(df, x_name, state_name, connect_at_transition=True):
    """
    Plotly can't handle gaps in a dataframe when splitting by state, so new columns should be created with explicit gaps
    where the state is off

    Note that this creates a large number of segments, which should be plotted using a loop

    Example:
        # Load data
        df = pd.DataFrame({'mode 0': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'mode 1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'mode 2': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # Get state information, with optional coloring. Note that the states will be an enum
        df['state'] = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True,
                                                                                  include_collision=False)
        ethogram_cmap = BehaviorCodes.ethogram_cmap(include_turns=True, include_reversal_turns=False)

        # Create modified dataframe using this function
        df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df, [0, 1, 2], 'state')

        # Loop to prep each line, then plot
        state_codes = df['state'].unique()
        phase_plot_list = []
        for i, state_code in enumerate(state_codes):
            phase_plot_list.append(
                        go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]], mode='lines',
                                     name=state_code.full_name, line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

        fig = go.Figure()
        fig.add_traces(phase_plot_list)
        fig.show()


    Note that this is not required for matplotlib; instead, nan values will be properly skipped
    See: plot_pca_projection_3d_from_project

    See:
    https://stackoverflow.com/questions/70407755/plotly-express-conditional-coloring-doesnt-work-properly/70408557#70408557

    Parameters
    ----------
    df
    x_name
    state_name: Should be the name of a BehaviorCodes enum or binary column
    connect_at_transition

    Returns
    -------

    """
    if isinstance(x_name, list):
        all_dfs_and_names = [modify_dataframe_to_allow_gaps_for_plotly(df, x, state_name) for x in x_name]
        all_dfs = [tmp[0] for tmp in all_dfs_and_names]
        all_names = [tmp[1] for tmp in all_dfs_and_names]
        df_concat = pd.concat(all_dfs, axis=1, ignore_index=False)
        return df_concat, all_names

    new_x_names = []
    new_columns = {}
    all_values = df[state_name].unique()
    for val in all_values:
        new_x_name = f"{x_name}-{val}"
        new_x_names.append(new_x_name)
        new_col = df[x_name].values.copy()
        nan_ind = df[state_name] != val
        new_col[nan_ind] = np.nan

        if connect_at_transition:
            starts, ends = get_contiguous_blocks_from_column(nan_ind, already_boolean=True)
            for s in starts:
                if s > 0:
                    new_col[s] = df[x_name].iat[s]
        new_columns[new_x_name] = new_col

    df_gaps = pd.DataFrame(new_columns)
    return df_gaps, new_x_names


def plot_with_shading(mean_vals, std_vals, xmax=None, x=None,
                      ax=None, std_vals_upper=None, show_legend=False, **kwargs):
    if std_vals_upper is not None:
        # Then the quantiles were passed, and they can be directly used
        upper_shading = std_vals_upper
        lower_shading = std_vals
    else:
        # Then the std was passed, and it must be added to the mean
        upper_shading = mean_vals + std_vals
        lower_shading = mean_vals - std_vals
    if ax is None:
        fig, ax = plt.subplots(dpi=100)
    if isinstance(mean_vals, pd.Series):
        x = list(mean_vals.index)
    else:
        if xmax is None:
            xmax = len(mean_vals)
        if x is None:
            x = np.arange(xmax)
        else:
            x = x[:xmax]
    # Main line
    ax.plot(x, mean_vals, **kwargs)
    # Shading
    fill_kwargs = {}
    if "color" in kwargs:
        fill_kwargs["color"] = kwargs["color"]
    ax.fill_between(x, upper_shading, lower_shading, alpha=0.2, linewidth=0.0, **fill_kwargs)
    if show_legend:
        ax.legend()
    return ax, lower_shading, upper_shading


def plot_with_shading_plotly(mean_vals, std_vals, xmax=None, fig=None, std_vals_upper=None, is_second_plot=False,
                             **kwargs):
    """
    Plot with shading, but for plotly. See plot_with_shading for matplotlib version

    See: https://plotly.com/python/continuous-error-bars/

    Parameters
    ----------
    mean_vals
    std_vals
    xmax
    ax
    std_vals_upper
    kwargs

    Returns
    -------

    """
    if std_vals_upper is not None:
        # Then the quantiles were passed, and they can be directly used
        upper_shading = std_vals_upper
        lower_shading = std_vals
    else:
        # Then the std was passed, and it must be added to the mean
        upper_shading = mean_vals + std_vals
        lower_shading = mean_vals - std_vals

    if isinstance(mean_vals, pd.Series):
        x = list(mean_vals.index)
    else:
        if xmax is None:
            xmax = len(mean_vals)
        x = np.arange(xmax)
    # Main line and shading together
    # Options for all lines
    i_line = 0 if not is_second_plot else 1
    color = kwargs.get('color', None)
    if color is None:
        color = px.colors.qualitative.Plotly[i_line]
    elif not isinstance(color, str):
        # Then it is matplotlib format, and we need to convert it
        color = float2rgba(color)
    opt = dict(
        x=x,
        line=dict(color=color),  # Need to specify the color so that the shading fill has the same color
    )

    if fig is None:
        fig = go.Figure()
        is_second_plot = False

    name = kwargs.get('label', f'Measurement_{i_line}')

    main_line = go.Scatter(
        name=name,
        y=mean_vals,
        mode='lines',
        **opt
    )
    # Options for only the shading lines
    color = main_line['line']['color']
    alpha = 0.2
    if 'rgb' in color:
        if 'rgba' not in color:
            fillcolor = color.replace('rgb', 'rgba').replace(')', f', {alpha})')
        else:
            # Reduce alpha
            split_color = color.split(',')
            split_color[-1] = f" {alpha})"
            fillcolor = ','.join(split_color)
    elif '#' in color:
        fillcolor = hex2rgba(color, alpha=alpha)
    else:
        raise ValueError(f"Unknown color format: {color}")

    opt_shading = dict(
        x=x,
        mode='lines',
        showlegend=False,
        fillcolor=fillcolor,
        line=dict(color='rgba(255,255,255,0)'),
    )
    # opt_shading.update(opt)

    shading_lines = [
        go.Scatter(
            name='Upper Bound',
            y=upper_shading,
            **opt_shading
        ),
        go.Scatter(
            name='Lower Bound',
            y=lower_shading,
            fill='tonexty',
            **opt_shading
        )
    ]

    fig.add_trace(main_line)
    fig.add_traces(shading_lines)

    return fig, lower_shading, upper_shading


def add_p_value_annotation(fig, array_columns=None, subplot=None, x_label=None, inner_x_label_pair=None,
                           _category_x_labels=None,
                           bonferroni_factor=None, height_mode='all_same',
                           _format=None, permutations=None, show_only_stars=False, show_ns=True,
                           separate_boxplot_fig=False, has_multicategory_index=False,
                           precalculated_p_values=None, DEBUG=False):
    """
    From: https://stackoverflow.com/questions/67505252/plotly-box-p-value-significant-annotation

    Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    Note: designed for individually adding traces using fig.add_trace, not plotly express
        However, does work with px.box with color using x_label='all'
        BUT: there must be an x label, with the colors producing paired boxes

    Example:
        fig = px.box(df, x="x", y="y", color="color")
        add_p_value_annotation(fig, x_label='all')

    Example using precalculated p-values:
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        func = lambda x: stats.ttest_1samp(x, 0)[1]
        df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
        df_pvalue = df_groupby['PC1 weight'].apply(func).to_frame()
        df_pvalue.columns = ['p_value']

        output = multipletests(df_pvalue.values.squeeze(), method='fdr_bh', alpha=0.05)
        df_pvalue['p_value_corrected'] = output[1]

        precalculated_p_values=df_significant_diff['p_value_corrected'].to_dict()

    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    x_label: None or str or 'all'
        if the boxplot has been separated by color, this specifies which color (x-axis label) to add the notation to
        In this case, array_columns should be the column numbers within a single label
    inner_x_label_pair: None or str
        if the x_label is multi-dimensional, this specifies the x_label to use for the p value comparison
    height_mode: str
        'all_same' (default) or 'top_of_data' (calculates the top data point and adds the annotation there)
    _format: dict
        format characteristics for the lines
    _all_x_labels: list
        list of all x_labels in the figure; only used when calculating x_labels via x_label='all'
    permutations: Optional[int]
        If not None, then do a non-parametric t-test using this many permutations
    separate_boxplot_fig: bool
        If True, then the figure contains separate boxplots for each color AND x_label, and searching for unique
        x_labels will fail if only fig.data[0] is used
    has_multicategory_index: bool
        If the figure is generated using multiple categories (see https://plotly.com/python/categorical-axes/#multicategorical-axes)
        then this splits the 2d index into two separate lists, used in different ways:
            i=0 is used for looping
            i=1 is used for indexing the y values and calculating the p value

    Returns:
    -------
    fig: figure
        figure with the added notation
    """
    if DEBUG:
        print(f"Adding p-value annotation to subplot {subplot}")

    if x_label == 'all':
        # Get all x_labels and call recursively
        if array_columns is not None:
            raise ValueError("If x_label is 'all', array_columns should be None")
        inner_x_label_pair = None
        if separate_boxplot_fig:
            all_x_labels_list = [fig.data[i].x for i in range(len(fig.data))]
            category_x_labels = pd.Series(np.concatenate(all_x_labels_list)).unique()
            # And we need to properly set the column indices
            array_columns_dict = defaultdict(list)
            for i, this_dat in enumerate(fig.data):
                # Check that there really is only one x_label in this data
                this_x_label = pd.Series(this_dat.x).unique()
                if len(this_x_label) > 1:
                    raise ValueError("The data contains multiple x_labels; use separate_boxplot_fig=False")
                this_x_label = this_x_label[0]
                array_columns_dict[this_x_label].append(i)
                assert len(array_columns_dict[this_x_label]) <= 2, "Only two columns can be compared"
        else:
            x_vec = np.array(fig.data[0].x)
            if x_vec.ndim >= 2:
                if not has_multicategory_index:
                    raise ValueError("If x label is multi-dimensional, then has_multicategory_index should be passed")
                # Get the inner (upper) x label
                inner_x_label_pair = pd.Series(x_vec[1, :]).unique()
                assert len(inner_x_label_pair) == 2, "Only two columns can be compared"
                # Get the outer labels, which will be different for each data entry list
                all_outer_labels = [pd.Series(d.x[0]).unique() for d in fig.data]
                try:
                    category_x_labels = pd.Series(np.squeeze(np.array(all_outer_labels))).unique()
                except ValueError:
                    # Occurs when there is only one outer label
                    category_x_labels = pd.Series(np.concatenate(all_outer_labels)).unique()
            else:
                category_x_labels = pd.Series(x_vec).unique()
            array_columns_dict = {x_label: None for x_label in category_x_labels}
        if DEBUG:
            print(f"Detected x_labels: {category_x_labels}")
        for x_label in category_x_labels:
            if x_label == 'all':
                logging.warning("x_label is 'all', which is a reserved keyword. Skipping")
                continue
            if bonferroni_factor is None:
                bonferroni_factor = len(category_x_labels)
            fig = add_p_value_annotation(fig, array_columns=[array_columns_dict[x_label]],
                                         subplot=subplot, x_label=x_label, show_ns=show_ns,
                                         _format=_format, _category_x_labels=category_x_labels, height_mode=height_mode,
                                         bonferroni_factor=bonferroni_factor, DEBUG=DEBUG, permutations=permutations,
                                         show_only_stars=show_only_stars, inner_x_label_pair=inner_x_label_pair,
                                         has_multicategory_index=has_multicategory_index)
        return fig

    if bonferroni_factor is None:
        bonferroni_factor = 1

    if array_columns is None or array_columns == [None]:
        if has_multicategory_index:
            # Then we need to compare the same column, indexed by the outer x label
            i = np.where(_category_x_labels == x_label)[0][0]
            array_columns = [[i, i]]
        else:
            array_columns = [[0, 1]]

    # Specify in what y_range to plot for each pair of columns
    default_text_format = dict(interline=0.07, text_height=1.07, color='black')
    if _format is None:
        _format = dict()
    _format = {**default_text_format, **_format}

    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01 + i * _format['interline'], 1.02 + i * _format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str = str(subplot)
        indices = []  # Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            # print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print(indices)
    else:
        subplot_str = ''

    if DEBUG:
        print(f"Testing columns: {array_columns}")

    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        # print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        # print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])
        y0 = np.array(fig_dict['data'][data_pair[0]]['y'])
        y1 = np.array(fig_dict['data'][data_pair[1]]['y'])

        if x_label is not None:
            # Then the x data also contains categories, and we should take a subset of y0 and y1 to match
            x0 = np.array(fig_dict['data'][data_pair[0]]['x'])
            x1 = np.array(fig_dict['data'][data_pair[1]]['x'])

            if x0.ndim >= 2:
                if not has_multicategory_index:
                    raise ValueError("If x label is multi-dimensional, then has_multicategory_index should be passed")
                x0 = x0[1, :]
                x1 = x1[1, :]

            if inner_x_label_pair is None:
                y0 = y0[np.where(x0 == x_label)[0]]
                y1 = y1[np.where(x1 == x_label)[0]]
            else:
                y0 = y0[np.where(x0 == inner_x_label_pair[0])[0]]
                y1 = y1[np.where(x1 == inner_x_label_pair[1])[0]]
            if DEBUG:
                print(f"y0: {y0[:5]}")
                print(f"y1: {y1[:5]}")
            # if DEBUG:
            #     print(f"y0: {y0}")
            #     print(f"y1: {y1}")
            # if len(y1) == 0:
            #     # Then the figure is organized per-color, and we can use the direct index
            #     y1 = fig_dict['data'][data_pair[1]]['y']

            # In addition, the x values of the annotation should be the same as the x_label, not the raw column number
            # First we need to get which x value the label corresponds to
            if _category_x_labels is None:
                category_x_labels = pd.Series(x0).unique()  # This keeps the order, unlike np.unique()
            else:
                category_x_labels = _category_x_labels
            x_label_ind = np.where(category_x_labels == x_label)[0][0]
            if has_multicategory_index:
                # Then each inner index will actually have its own x value, i.e. the outer index spans 2 columns
                # i.e. 0 should map to 0.5, and 1 should map to 2.5
                x_label_ind = x_label_ind * 2 + 0.5
            column_pair = [x_label_ind - 0.2, x_label_ind + 0.2]

        # Drop any nan values
        y0 = y0[~np.isnan(y0)]
        y1 = y1[~np.isnan(y1)]

        if len(y0) == 0 or len(y1) == 0:
            print(f"Skipping annotation for {x_label} because one of the datasets is empty")
            continue

        # Get the p-value
        if precalculated_p_values is None:
            pvalue = stats.ttest_ind(y0, y1, equal_var=False, random_state=4242, permutations=permutations)[1] * bonferroni_factor
        else:
            pvalue = precalculated_p_values[x_label]
        if DEBUG:
            print(f"p-value: {pvalue} for x_label {x_label}")
        significance_stars = p_value_to_stars(pvalue)

        if not show_ns and significance_stars == 'ns':
            continue

        # Get the y value to plot the annotation
        y_range_of_plot = np.max(y_range[index])
        if height_mode == 'all_same':
            annotation_y_shift = -y_range_of_plot * 0.1  # Shift annotation down by this amount
            y0_annotation = y_range[index][0] + annotation_y_shift
            y1_annotation = y_range[index][1] + annotation_y_shift
            y_ref = "y" + subplot_str + " domain"
        elif height_mode == 'top_of_data':
            annotation_y_shift = y_range_of_plot * 0.01  # Shift annotation up by this amount
            y0_annotation = np.max(y0) + annotation_y_shift
            y1_annotation = np.max(y1) + annotation_y_shift
            y_ref = "y"
        else:
            raise ValueError(f"Unknown height_mode: {height_mode}")

        # Actually plot the annotation
        if not show_only_stars:
            # Vertical line
            fig.add_shape(type="line",
                          xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                          x0=column_pair[0], y0=y0_annotation,
                          x1=column_pair[0], y1=y1_annotation,
                          line=dict(color=_format['color'], width=2, )
                          )
            # Horizontal line
            fig.add_shape(type="line",
                          xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                          x0=column_pair[0], y0=y0_annotation,
                          x1=column_pair[1], y1=y1_annotation,
                          line=dict(color=_format['color'], width=2, )
                          )
            # Vertical line
            fig.add_shape(type="line",
                          xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                          # x0=column_pair[1], y0=y_range[index][0] + 1.5*annotation_y_shift,
                          x0=column_pair[1], y0=y0_annotation,
                          x1=column_pair[1], y1=y1_annotation,
                          line=dict(color=_format['color'], width=2, )
                          )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'], size=14),
                                x=(column_pair[0] + column_pair[1]) / 2,
                                # y=y_range[index][1] * _format['text_height'] + annotation_y_shift,
                                y=np.max([y0_annotation, y1_annotation]),
                                showarrow=False,
                                text=significance_stars,
                                textangle=0,
                                xref="x" + subplot_str,
                                yref=y_ref
                                ))
        if DEBUG:
            print(f"p-value: {pvalue} for x_label {x_label}")
            print(f"Adding annotation at x={column_pair[0]} and {column_pair[1]}")
            print(f"Adding annotation at y={y0_annotation} and {y1_annotation}")
            # err
    return fig


def p_value_to_stars(pvalue):
    if pvalue >= 0.05:
        significance_stars = 'ns'
    elif pvalue >= 0.01:
        significance_stars = '*'
    elif pvalue >= 0.001:
        significance_stars = '**'
    else:
        significance_stars = '***'
    return significance_stars


# Do a set of ID-able neurons triggered to reverse, then forward

# First, do wbfm

def plot_triggered_averages(project_data_list, output_foldername=None,
                            project_data_color_map=None):
    """
    For figure 1 of the paper: averaged triggered averages between two example datasets

    Parameters
    ----------
    project_data_list
    to_save

    Returns
    -------

    """

    from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
    xlim = [-5, 15]
    trace_opt = paper_trace_settings()
    if project_data_color_map is None:
        base_cmap = plotly_paper_color_discrete_map()
        project_data_color_map = [base_cmap['wbfm'], base_cmap['immob']]

    # Example neurons
    for neuron_base in ['AVAL', 'PC1']:
        for state in [BehaviorCodes.REV, BehaviorCodes.FWD]:

            fig, ax = plt.subplots(dpi=200, figsize=(4, 2.5))
            for i_trace, project_data in enumerate(project_data_list):

                color = project_data_color_map[i_trace]
                df_traces = project_data.calc_paper_traces()
                ind_class = project_data.worm_posture_class.calc_triggered_average_indices(state=state)

                # Actually plot
                if neuron_base == "PC1":
                    y, _ = project_data.calc_pca_modes(2, **trace_opt, multiply_by_variance=True)
                    y = pd.Series(y.loc[:, 0])
                    neuron = neuron_base
                else:
                    neuron = neuron_base
                    if neuron not in df_traces.columns:
                        continue
                    y = df_traces[neuron]

                mat = ind_class.calc_triggered_average_matrix(y)

                ind_class.plot_triggered_average_from_matrix(mat, ax=ax, is_second_plot=(i_trace > 0),
                                                             lw=2, color=color)
                ax.set_title(f"{neuron}")
                ax.set_xlim(xlim)
                ax.set_xlabel(project_data.x_label_for_plots)
                ax.set_ylabel("dR/R20")
                plt.tight_layout()

                # Make a fake behavioral vector to shade this
                if state == BehaviorCodes.REV:
                    behavior_shading_type = 'rev'
                else:
                    behavior_shading_type = 'fwd'
                from wbfm.utils.general.utils_behavior_annotation import add_behavior_shading_to_plot
                add_behavior_shading_to_plot(ind_class.ind_preceding, mat.columns, behavior_shading_type, ax,
                                             DEBUG=False)

            apply_figure_settings(fig, width_factor=0.3, height_factor=0.15, plotly_not_matplotlib=False)

            # Save
            if output_foldername is not None:
                fname = f"example-{neuron}-{state}.png"
                # if 'immob' in fname:
                #     fname = os.path.join(output_foldername, 'immob', fname)
                # else:
                #     fname = os.path.join(output_foldername, 'gcamp', fname)
                fname = os.path.join(output_foldername, fname)
                plt.savefig(fname)
                fname = Path(fname).with_suffix('.svg')
                plt.savefig(fname)


def convert_channel_mode_to_axis_label(channel_mode):
    if isinstance(channel_mode, dict):
        channel_mode = channel_mode.get('channel_mode', 'dr_over_r_50')
    elif channel_mode is None:
        channel_mode = 'dr_over_r_50'  # Default

    if channel_mode == 'dr_over_r_20':
        return r"$\Delta R/R_{20}$"
    elif channel_mode == 'dr_over_r_50':
        return r"$\Delta R/R_{50}$"
    else:
        raise ValueError(f"Unknown channel mode: {channel_mode}")
