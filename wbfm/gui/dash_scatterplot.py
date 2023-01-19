from pathlib import Path

from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd

from wbfm.utils.external.utils_pandas import correlate_return_cross_terms
from wbfm.utils.projects.project_export import read_dataframes_from_exported_folder
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


def build_wbfm_dashboard(project_path):

    app = Dash(__name__)
    dict_of_dataframes = read_dataframes_from_exported_folder(project_path)
    df_behavior = dict_of_dataframes['behavior']['behavior']
    df_curvature = dict_of_dataframes['behavior']['curvature']
    dict_of_traces_dfs = dict_of_dataframes['traces']
    # _df_traces = dict_of_dataframes['traces']['ratio']

    # Add a time column
    # df_time = df_behavior.index
    # dict_of_traces_dfs = {k: df.reset_index().rename(columns={'index': 'time'}, copy=False)
    #                       for k, df in dict_of_traces_dfs.items()}
    # df_behavior = df_behavior.reset_index().rename(columns={'index': 'time'}, copy=False)
    # df_all_time_series = pd.concat([_df_behavior, _df_traces, _df_curvature], axis=1).reset_index()
    # df_all_time_series.rename(columns={'index': 'time'}, inplace=True, copy=False)

    # Define layout
    app.layout = html.Div([
        build_dropdowns(df_behavior, df_curvature, dict_of_traces_dfs),
        build_second_row_options(),
        build_plots_html(),
        build_plots_curvature(df_curvature)
        ]
    )

    # Main scatter plot changing callback (change axes)
    @app.callback(
        Output('correlation-scatterplot', 'figure'),
        Input('scatter-xaxis', 'value'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value')
    )
    def _update_scatter_plot(x_name, y_name, neuron_name, regression_type, trace_type):
        df_traces = dict_of_traces_dfs[trace_type]
        return update_scatter_plot(df_behavior, df_traces, x_name, y_name, neuron_name, regression_type)

    # Neuron selection updates
    # Logic: everything goes through the dropdown menu. A click will update that, which updates other things
    @app.callback(
        Output('neuron-select-dropdown', 'value'),
        Output('correlation-scatterplot', 'clickData'),
        Output('kymograph-all-neuron-max-segment-correlation', 'clickData'),
        Input('correlation-scatterplot', 'clickData'),
        Input('kymograph-all-neuron-max-segment-correlation', 'clickData')
    )
    def _use_click_to_update_neuron_dropdown(correlation_clickData, kymograph_clickData):
        # Resets the clickData of each plot (multiple outputs)
        if correlation_clickData:
            neuron_name = correlation_clickData["points"][0]["customdata"][0]
        elif kymograph_clickData:
            neuron_name = kymograph_clickData["points"][0]["customdata"][0]
        else:
            neuron_name = None
        return neuron_name, None, None

    @app.callback(
        Output('kymograph-select-dropdown', 'value'),
        Input('kymograph-per-segment-correlation', 'clickData')
    )
    def _use_click_to_update_kymograph_segment(clickData):
        # print(clickData)
        if clickData is None:
            return 'segment_001'
        kymograph_segment_name = clickData["points"][0]["x"]
        return kymograph_segment_name

    @app.callback(
        Output('neuron-trace', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value')
    )
    def _update_neuron_trace(neuron_name, regression_type, trace_type):
        df_traces = dict_of_traces_dfs[trace_type]
        df_behavior_and_neurons = pd.concat([df_behavior, df_traces], axis=1)
        return update_neuron_trace_plot(df_behavior_and_neurons, neuron_name, regression_type)

    @app.callback(
        Output('trace-and-behavior-scatterplot', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value')
    )
    def _update_behavior_scatter(neuron_name, behavior_name, regression_type, trace_type):
        df_traces = dict_of_traces_dfs[trace_type]
        df_behavior_and_neurons = pd.concat([df_behavior, df_traces], axis=1)
        return update_behavior_scatter_plot(df_behavior_and_neurons, behavior_name, neuron_name, regression_type)

    # Behavior updates
    @app.callback(
        Output('behavior-trace', 'figure'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('regression-type', 'value')
    )
    def _update_behavior_trace(behavior_name, regression_type):
        return update_behavior_trace_plot(df_behavior, behavior_name, regression_type)

    @app.callback(
        Output('kymograph-scatter', 'figure'),
        Input('kymograph-select-dropdown', 'value'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value')
    )
    def _update_kymograph_scatter(kymograph_segment_name, neuron_name, regression_type, trace_type):
        df_traces = dict_of_traces_dfs[trace_type]
        df_all = pd.concat([df_behavior, df_curvature, df_traces], axis=1)
        return update_kymograph_scatter_plot(df_all, kymograph_segment_name, neuron_name, regression_type)

    @app.callback(
        Output('kymograph-per-segment-correlation', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value')
    )
    def _update_kymograph_correlation(neuron_name, regression_type, trace_type):
        df_traces = dict_of_traces_dfs[trace_type]
        return update_kymograph_correlation_per_segment(df_traces, df_behavior, df_curvature, neuron_name,
                                                        regression_type)

    @app.callback(
        Output('kymograph-all-neuron-max-segment-correlation', 'figure'),
        Input('regression-type', 'value'),
        Input('trace-select-dropdown', 'value'),
        Input('kymograph-range-slider', 'value')
    )
    def _update_kymograph_max_segment(regression_type, trace_type, kymograph_range):
        df_traces = dict_of_traces_dfs[trace_type]
        return update_max_correlation_over_all_segment_plot(df_behavior, df_traces, df_curvature, regression_type,
                                                            kymograph_range)

    if __name__ == '__main__':
        app.run_server(debug=True)


def update_scatter_plot(df_behavior, df_traces, x_name, y_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        y_corr_rev = df_traces.corrwith(df_behavior[y_name][rev_idx])
        y_corr_fwd = df_traces.corrwith(df_behavior[y_name][~rev_idx])
        x_corr = df_traces.corrwith(df_behavior[x_name])
        # x_corr_rev = df_traces.corrwith(df_all_time_series[x_name][rev_idx])
        # x_corr_fwd = df_traces.corrwith(df_all_time_series[x_name][~rev_idx])
        # Combine in a dataframe for plotting
        # df_dict = {'y_rev': y_corr_rev, 'y_fwd': y_corr_fwd, 'x_rev': x_corr_rev, 'x_fwd': x_corr_fwd}
        df_dict = {'y_rev': y_corr_rev, 'y_fwd': y_corr_fwd, x_name: x_corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['y_fwd', 'y_rev']
        # x_names = ['x_fwd', 'x_rev']
    else:
        y_corr = df_traces.corrwith(df_behavior[y_name])
        x_corr = df_traces.corrwith(df_behavior[x_name])
        # Combine in a dataframe for plotting
        df_dict = {y_name: y_corr, x_name: x_corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = y_name
    x_name = x_name
    df_corr['neuron_name'] = get_names_from_df(df_traces)
    # Create a fake "selected" column, because the default doesn't work when you return a new figure
    df_corr['selected'] = 1
    df_corr.loc[neuron_name, 'selected'] = 5
    # Top scatter plot
    _fig = px.scatter(df_corr, x=x_name, y=y_names, hover_name="neuron_name",
                      custom_data=["neuron_name"],
                      size='selected',
                      marginal_y='histogram',
                      title="Are neurons correlated to 2 behaviors?",
                      trendline="ols")
    # _fig.update_layout(clickmode='event+select')
    # selected_points = [int(neuron_name.split('_')[1]) - 1]
    # print(df_corr, selected_points)
    # _fig.update_traces(selectedpoints=[selected_points])
    # Half as tall
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_neuron_trace_plot(df_behavior_and_neurons, neuron_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_behavior_and_neurons, x=df_behavior_and_neurons.index, y=neuron_name,
                   title=f"Trace for {neuron_name}",
                   range_x=[0, df_behavior_and_neurons.shape[0]], **opt)
    _fig.update_layout(height=325, margin={'l': 40, 'b': 40, 't': 30, 'r': 0})
    return _fig


def update_behavior_scatter_plot(df_behavior_and_neurons, behavior_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_behavior_and_neurons, x=behavior_name, y=neuron_name,
                      title=f"Is {neuron_name} correlated to behavior?",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_scatter_plot(df_all, kymograph_segment_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_all, x=kymograph_segment_name, y=neuron_name,
                      title=f"Is {neuron_name} correlated to behavior?",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_correlation_per_segment(df_traces, df_behavior, df_curvature, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        corr_rev = df_curvature.corrwith(df_traces[neuron_name][rev_idx])
        corr_fwd = df_curvature.corrwith(df_traces[neuron_name][~rev_idx])
        # Combine in a dataframe for plotting
        df_dict = {'rev': corr_rev, 'fwd': corr_fwd}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['fwd', 'rev']
    else:
        corr = df_curvature.corrwith(df_traces[neuron_name])
        # Combine in a dataframe for plotting
        df_dict = {'correlation': corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['correlation']
    _fig = px.line(df_corr, y=y_names, title=f"Which body segment correlates to {neuron_name}?", range_y=[-0.8, 0.8])
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_max_correlation_over_all_segment_plot(df_behavior, df_traces, df_curvature, regression_type,
                                                 kymograph_range):
    # Will not actually be updated, except for changing the rectification
    df_curvature_subset = df_curvature.iloc[:, kymograph_range[0]:kymograph_range[1]]
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        df_corr = correlate_return_cross_terms(df_traces[rev_idx], df_curvature_subset)
        df_max_rev = df_corr.abs().max(axis=1)

        df_corr = correlate_return_cross_terms(df_traces[~rev_idx], df_curvature_subset)
        df_max_fwd = df_corr.abs().max(axis=1)

        df_dict = {'rev': df_max_rev, 'fwd': df_max_fwd}
        df_corr_max = pd.DataFrame(df_dict)
        y_names = ['fwd', 'rev']
    else:
        df_corr = correlate_return_cross_terms(df_traces, df_curvature_subset)
        df_corr_max = pd.DataFrame(df_corr.max(axis=1), columns=['correlation'])
        y_names = 'correlation'
    # For setting custom data
    df_corr_max['index'] = df_corr_max.index

    _fig = px.scatter(df_corr_max, y=y_names, title=f"Max curvature correlation over selected body segments", range_y=[0, 0.8],
                      marginal_y='histogram', custom_data=['index'])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_behavior_trace_plot(df_behavior, behavior_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_behavior, x=df_behavior.index, y=behavior_name,
                   range_x=[0, len(df_behavior)],
                   title=f"Trace of {behavior_name}", **opt)
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def switch_plot_func_using_rectification(regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
        px_func = px.scatter
    else:
        opt = {}
        px_func = px.line
    return opt, px_func


def build_dropdowns(df_behavior, df_curvature, dict_of_trace_dataframes) -> html.Div:
    curvature_names = get_names_from_df(df_curvature)
    curvature_initial = curvature_names[0]
    trace_names = list(dict_of_trace_dataframes.keys())
    trace_initial = 'ratio'
    df_traces = dict_of_trace_dataframes[trace_initial]
    neuron_names = get_names_from_df(df_traces)
    neuron_initial = neuron_names[0]

    behavior_names = get_names_from_df(df_behavior)

    x_initial = 'signed_speed_angular'
    behavior_initial = 'signed_middle_body_speed'

    header = html.H1("Dropdowns for changing all plots")

    dropdown_style = {'width': '24%'}
    dropdowns = html.Div([
        html.Div([
            html.Label(["Behavior to correlate (x axis)"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    x_initial,
                    id='scatter-xaxis',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Behavior to show and correlate (y axis)"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    behavior_initial,
                    id='behavior-scatter-yaxis',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select neuron"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    neuron_names,
                    neuron_initial,
                    id='neuron-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select kymograph segment"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    curvature_names,
                    curvature_initial,
                    id='kymograph-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select trace type"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    trace_names,
                    trace_initial,
                    id='trace-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style)
        ], style={'display': 'flex', 'padding': '10px 5px'})

    # return html.Div(dropdowns, style={'display': 'flex', 'padding': '10px 5px'})
    return html.Div([header, dropdowns])


def build_second_row_options() -> html.Div:

    row_style = {'display': 'inline-block', 'width': '50%'}

    return html.Div([
        html.Div([
            html.Label(['Style of regression line'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.RadioItems(
                ['Overall regression', 'Rectified regression'],
                'Overall regression',
                id='regression-type'),
        ], style=row_style),

        html.Div([
            html.Label(['Range for kymograph max plot'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.RangeSlider(0, 99, 3, value=[5, 95], id='kymograph-range-slider', allowCross=False)
        ], style=row_style)
        ]
    )


def build_plots_html() -> html.Div:
    # Second trace plot, which is actually initialized through a clickData field on the scatterplot
    initial_neuron_clickData = {'points': [{'customdata': ['neuron_001']}]}

    top_header = html.H2("Summary plots (some interactive)")

    top_row_style = {'width': '20%', 'display': 'inline-block'}
    top_row = html.Div([
        html.Div([dcc.Graph(id='correlation-scatterplot', clickData=initial_neuron_clickData)], style=top_row_style),
        html.Div([dcc.Graph(id='trace-and-behavior-scatterplot')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-scatter')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-per-segment-correlation')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-all-neuron-max-segment-correlation',
                            clickData=initial_neuron_clickData)], style=top_row_style)
    ], style={'width': '100%', 'display': 'inline-block'})

    time_series_header = html.H2("Time Series plots")
    time_series_rows = html.Div([
        dcc.Graph(id='neuron-trace'),
        dcc.Graph(id='behavior-trace')
    ], style={'width': '100%', 'display': 'inline-block'}
    )

    return html.Div([top_header, top_row, time_series_header, time_series_rows])


def build_plots_curvature(df_curvature) -> html.Div:
    fig = px.imshow(df_curvature.T, zmin=-0.05, zmax=0.05, aspect=3, color_continuous_scale='RdBu')

    image = html.Div([
        dcc.Graph(id='kymograph-image', figure=fig)
    ], style={'width': '100%', 'display': 'inline-block'})
    return image


if __name__ == "__main__":

    # DATA_FOLDER = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/tmp_data"
    project_path = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
    build_wbfm_dashboard(project_path)
