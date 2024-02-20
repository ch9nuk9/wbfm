import logging
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.linear_model import LinearRegression

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan, get_contiguous_blocks_from_column
from wbfm.utils.general.utils_paper import paper_trace_settings, apply_figure_settings
from wbfm.utils.traces.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
import plotly.graph_objects as go


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


def plot_with_shading_plotly(mean_vals, std_vals, xmax=None, fig=None, std_vals_upper=None, **kwargs):
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

    if xmax is None:
        xmax = len(mean_vals)
    x = np.arange(xmax)
    # Main line and shading together
    # Options for all lines
    opt = dict(
        mode='lines',
        x=x,
        line=dict(color='rgb(31, 119, 180)'),
    )

    if fig is None:
        fig = go.Figure()

    main_line = go.Scatter(
        name='Measurement',
        y=mean_vals,
        **opt
    )
    # Options for only the shading lines
    fillcolor = main_line['line']['color'].replace('rgb', 'rgba').replace(')', ', 0.5)')
    opt_shading = dict(
        line=dict(width=0),
        showlegend=False,
        marker=dict(color="#444"),
        fillcolor=fillcolor,
        opacity=0.2,
    )
    opt_shading.update(opt)

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


def add_p_value_annotation(fig, array_columns=None, subplot=None, x_label=None, bonferroni_factor=None,
                           _format=None, permutations=None, DEBUG=False):
    """
    From: https://stackoverflow.com/questions/67505252/plotly-box-p-value-significant-annotation

    Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    Note: designed for individually adding traces using fig.add_trace, not plotly express
        However, does work with px.box with color using x_label='all'
        BUT: there must be an x label, with the colors producing paired boxes

    Example:
        fig = px.box(df, x="x", y="y", color="color")
        add_p_value_annotation(fig, x_label='all')

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
    _format: dict
        format characteristics for the lines
    permutations: Optional[int]
        If not None, then do a non-parametric t-test using this many permutations

    Returns:
    -------
    fig: figure
        figure with the added notation
    """
    if DEBUG:
        print(f"Adding p-value annotation to subplot {subplot}")

    if x_label == 'all':
        # Get all x_labels and call recursively
        all_x_labels = pd.Series(fig.data[0].x).unique()
        if DEBUG:
            print(f"Detected x_labels: {all_x_labels}")
        for x_label in all_x_labels:
            if x_label == 'all':
                logging.warning("x_label is 'all', which is a reserved keyword. Skipping")
                continue
            if bonferroni_factor is None:
                bonferroni_factor = len(all_x_labels)
            fig = add_p_value_annotation(fig, array_columns, subplot=subplot, x_label=x_label, _format=_format,
                                         bonferroni_factor=bonferroni_factor, DEBUG=DEBUG, permutations=permutations)
        return fig

    if array_columns is None:
        array_columns = [[0, 1]]

    annotation_y_shift = -0.08  # Shift annotation down by this amount

    # Specify in what y_range to plot for each pair of columns
    if _format is None:
        _format = dict(interline=0.07, text_height=1.07, color='black')
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

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        # print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        # print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])
        y0 = fig_dict['data'][data_pair[0]]['y']
        y1 = fig_dict['data'][data_pair[1]]['y']

        if x_label is not None:
            # Then the x data also contains categories, and we should take a subset of y0 and y1 to match
            x0 = fig_dict['data'][data_pair[0]]['x']
            x1 = fig_dict['data'][data_pair[1]]['x']

            y0 = y0[np.where(x0 == x_label)[0]]
            y1 = y1[np.where(x1 == x_label)[0]]
            # if len(y1) == 0:
            #     # Then the figure is organized per-color, and we can use the direct index
            #     y1 = fig_dict['data'][data_pair[1]]['y']
            if DEBUG:
                print(y0)
                print(y1)

            # In addition, the x values of the annotation should be the same as the x_label, not the raw column number
            # First we need to get which x value the label corresponds to
            all_x_labels = pd.Series(x0).unique()  # This keeps the order, unlike np.unique()
            x_label_ind = np.where(all_x_labels == x_label)[0][0]
            column_pair = [x_label_ind - 0.2, x_label_ind + 0.2]

        # Drop any nan values
        valid_idx = np.logical_and(~np.isnan(y0), ~np.isnan(y1))
        y0 = y0[valid_idx]
        y1 = y1[valid_idx]

        # Get the p-value
        pvalue = stats.ttest_ind(y0, y1, equal_var=False, random_state=4242, permutations=permutations)[1] * bonferroni_factor
        # if DEBUG:
        #     print(f"p-value: {pvalue}")
        #     print(f"Data: {y0}, {y1}")
        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01:
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][0] + 1.5*annotation_y_shift,
                      x1=column_pair[0], y1=y_range[index][1] + 1.5*annotation_y_shift,
                      line=dict(color=_format['color'], width=2, )
                      )
        # Horizontal line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][1] + 1.5*annotation_y_shift,
                      x1=column_pair[1], y1=y_range[index][1] + 1.5*annotation_y_shift,
                      line=dict(color=_format['color'], width=2, )
                      )
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[1], y0=y_range[index][0] + 1.5*annotation_y_shift,
                      x1=column_pair[1], y1=y_range[index][1] + 1.5*annotation_y_shift,
                      line=dict(color=_format['color'], width=2, )
                      )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'], size=14),
                                x=(column_pair[0] + column_pair[1]) / 2,
                                y=y_range[index][1] * _format['text_height'] + annotation_y_shift,
                                showarrow=False,
                                text=symbol,
                                textangle=0,
                                xref="x" + subplot_str,
                                yref="y" + subplot_str + " domain"
                                ))
        if DEBUG:
            print(f"p-value: {pvalue}")
            print(f"Adding annotation at x= {column_pair[0]} and {column_pair[1]}")
            print(f"Adding annotation at y={y_range[index][1] * _format['text_height'] + annotation_y_shift}")
    return fig


# Do a set of ID-able neurons triggered to reverse, then forward

# First, do wbfm

def plot_triggered_averages(project_data_list, output_foldername=None):
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

    # Example neurons
    for neuron_base in ['AVAL', 'PC1']:
        # for neuron_base in ['AVA', 'RME', 'VB02', 'BAG', 'PC1']:
        for state in [BehaviorCodes.REV, BehaviorCodes.FWD]:

            fig, ax = plt.subplots(dpi=200, figsize=(4, 2.5))
            for i_trace, project_data in enumerate(project_data_list):

                df_traces = project_data.calc_paper_traces()
                ind_class = project_data.worm_posture_class.calc_triggered_average_indices(state=state)

                # Actually plot
                if neuron_base == "PC1":
                    y = project_data.calc_pca_modes(2, **trace_opt, multiply_by_variance=True)
                    y = pd.Series(y.loc[:, 0])
                    neuron = neuron_base
                else:
                    neuron = neuron_base
                    if neuron not in df_traces.columns:
                        continue
                    y = df_traces[neuron]

                mat = ind_class.calc_triggered_average_matrix(y)

                ind_class.plot_triggered_average_from_matrix(mat, ax=ax, is_second_plot=(i_trace > 0), lw=2)
                ax.set_title(f"{neuron}")
                ax.set_xlim(xlim)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Activity")
                plt.tight_layout()

                # Make a fake behavioral vector to shade this
                if state == BehaviorCodes.REV:
                    behavior_shading_type = 'rev'
                else:
                    behavior_shading_type = 'fwd'
                from wbfm.utils.general.utils_behavior_annotation import shade_triggered_average
                shade_triggered_average(ind_class.ind_preceding, mat.columns, behavior_shading_type, ax,
                                        DEBUG=False)

            apply_figure_settings(fig, width_factor=0.3, height_factor=0.15, plotly_not_matplotlib=False)

            # Save
            if output_foldername is not None:
                fname = f"multiproject-{neuron}-{state}.png"
                # if 'immob' in fname:
                #     fname = os.path.join(output_foldername, 'immob', fname)
                # else:
                #     fname = os.path.join(output_foldername, 'gcamp', fname)
                fname = os.path.join(output_foldername, fname)
                plt.savefig(fname)
                fname = Path(fname).with_suffix('.svg')
                plt.savefig(fname)
