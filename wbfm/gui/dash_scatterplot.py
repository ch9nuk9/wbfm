from pathlib import Path
from typing import List

import numpy as np
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd

from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df

DATA_FOLDER = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/tmp_data"


def main():

    app = Dash(__name__)
    for file in Path(DATA_FOLDER).iterdir():
        if 'df_traces' in file.name:
            _df_traces = pd.read_hdf(file)
        elif 'df_correlation' in file.name:
            df_correlation = pd.read_hdf(file)
        elif 'df_behavior' in file.name:
            _df_behavior = pd.read_hdf(file)
        elif 'df_curvature' in file.name:
            _df_curvature = pd.read_hdf(file)

    # Combine everything, including an index column
    df_all_time_series = pd.concat([_df_behavior, _df_traces, _df_curvature], axis=1).reset_index()
    df_all_time_series.rename(columns={'index': 'time'}, inplace=True, copy=False)

    # Define layout
    app.layout = html.Div([
        build_dropdowns(df_correlation, _df_behavior, _df_traces, _df_curvature),
        build_regression_menu(),
        build_plots_html(),
        build_plots_curvature(_df_curvature)
        ]
    )

    # Main scatter plot changing callback (change axes)
    @app.callback(
        Output('correlation-scatterplot', 'figure'),
        Input('scatter-xaxis', 'value'),
        Input('scatter-yaxis', 'value')
    )
    def _update_scatter_plot(x_name, y_name):
        _fig = update_scatter_plot(df_correlation, x_name, y_name)
        return _fig

    # Neuron selection updates
    # Logic: everything goes through the dropdown menu. A click will update that, which updates other things
    @app.callback(
        Output('neuron-select-dropdown', 'value'),
        Input('correlation-scatterplot', 'clickData')
    )
    def _use_click_to_update_dropdown(clickData):
        neuron_name = clickData["points"][0]["customdata"][0]
        return neuron_name

    @app.callback(
        Output('neuron-trace', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value')
    )
    def _update_neuron_trace(neuron_name, regression_type):
        return update_neuron_trace_plot(df_all_time_series, neuron_name, regression_type)

    @app.callback(
        Output('trace-and-behavior-scatterplot', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('regression-type', 'value')
    )
    def _update_behavior_scatter(neuron_name, behavior_name, regression_type):
        return update_behavior_scatter_plot(df_all_time_series, behavior_name, neuron_name, regression_type)

    # Behavior updates
    @app.callback(
        Output('behavior-trace', 'figure'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('regression-type', 'value')
    )
    def _update_behavior_trace(behavior_name, regression_type):
        return update_behavior_trace_plot(df_all_time_series, behavior_name, regression_type)

    @app.callback(
        Output('kymograph-scatter', 'figure'),
        Input('kymograph-select-dropdown', 'value'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value')
    )
    def _update_kymograph_scatter(kymograph_segment_name, neuron_name, regression_type):
        return update_kymograph_scatter_plot(df_all_time_series, kymograph_segment_name, neuron_name, regression_type)

    @app.callback(
        Output('kymograph-per-segment-correlation', 'figure'),
        Input('neuron-select-dropdown', 'value'),
        Input('regression-type', 'value')
    )
    def _update_kymograph_correlation(neuron_name, regression_type):
        return update_kymograph_correlation_per_segment(df_all_time_series, _df_curvature, neuron_name, regression_type)

    if __name__ == '__main__':
        app.run_server(debug=True)


def build_plots_html() -> html.Div:
    # Second trace plot, which is actually initialized through a clickData field on the scatterplot
    initial_clickData = {'points': [{'customdata': ['neuron_001']}]}

    top_row_style = {'width': '25%', 'display': 'inline-block'}

    top_row = html.Div([
        html.Div([dcc.Graph(id='correlation-scatterplot', clickData=initial_clickData)], style=top_row_style),
        html.Div([dcc.Graph(id='trace-and-behavior-scatterplot')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-scatter')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-per-segment-correlation')], style=top_row_style)
    ], style={'width': '100%', 'display': 'inline-block'})

    additional_rows = html.Div([
        dcc.Graph(id='neuron-trace'),
        dcc.Graph(id='behavior-trace')
    ], style={'width': '100%', 'display': 'inline-block'}
    )

    return html.Div([top_row, additional_rows])


def build_plots_curvature(df_curvature) -> html.Div:
    fig = px.imshow(df_curvature.T, zmin=-0.05, zmax=0.05, aspect=3, color_continuous_scale='RdBu')

    image = html.Div([
        dcc.Graph(id='kymograph-image', figure=fig)
    ], style={'width': '100%', 'display': 'inline-block'})
    return image


def update_scatter_plot(df_correlation, x_name, y_name):
    # Top scatter plot
    _fig = px.scatter(df_correlation, x=x_name, y=y_name, hover_name="neuron_name",
                      custom_data=["neuron_name"],
                      marginal_y='histogram',
                      title="Interactive Scatterplot")
    # Half as tall
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_neuron_trace_plot(df_all_time_series, neuron_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_all_time_series, x='time', y=neuron_name,
                   title=f"Trace for {neuron_name}",
                   range_x=[0, len(df_all_time_series)], **opt)
    _fig.update_layout(height=325, margin={'l': 40, 'b': 40, 't': 30, 'r': 0})
    return _fig


def update_behavior_scatter_plot(df_all_time_series, behavior_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_all_time_series, x=behavior_name, y=neuron_name,
                      title=f"Behavior-neuron scatterplot",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    results = px.get_trendline_results(_fig)
    print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_scatter_plot(df_all_time_series, kymograph_segment_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_all_time_series, x=kymograph_segment_name, y=neuron_name,
                      title=f"Kymograph-neuron scatterplot",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_correlation_per_segment(df_all_time_series, df_curvature, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        rev_idx = df_all_time_series.reversal
        corr_rev = df_curvature.corrwith(df_all_time_series[neuron_name][rev_idx])
        corr_fwd = df_curvature.corrwith(df_all_time_series[neuron_name][~rev_idx])
        # Combine in a dataframe for plotting
        df_dict = {'rev': corr_rev, 'fwd': corr_fwd}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['rev', 'fwd']
    else:
        corr = df_curvature.corrwith(df_all_time_series[neuron_name])
        # Combine in a dataframe for plotting
        df_dict = {'correlation': corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['correlation']
    _fig = px.line(df_corr, y=y_names, title=f"Per-segment correlation", range_y=[-0.8, 0.8])
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_behavior_trace_plot(df_all_time_series, behavior_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_all_time_series, x='time', y=behavior_name,
                   range_x=[0, len(df_all_time_series)],
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


def build_dropdowns(df_correlation, df_behavior, df_traces, df_curvature) -> html.Div:
    curvature_names = get_names_from_df(df_curvature)
    curvature_initial = curvature_names[0]
    neuron_names = get_names_from_df(df_traces)
    neuron_initial = neuron_names[0]
    correlation_names = get_names_from_df(df_correlation)
    correlation_names.remove('neuron_name')
    correlation_names_no_dummy = correlation_names.copy()
    correlation_names_no_dummy.remove('dummy')

    behavior_names = get_names_from_df(df_behavior)

    y_initial = 'coefficient'
    x_initial = 'dummy'
    behavior_initial = 'signed_speed'

    style = {'width': '24%'}

    return html.Div(children=[
        html.Div([
            html.Label(['Scatter y axis'], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    correlation_names_no_dummy,
                    y_initial,
                    id='scatter-yaxis',
                    clearable=False
                ),
            ])],
            style=style),

        html.Div([
            html.Label(['Scatter x axis'], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    correlation_names,
                    x_initial,
                    id='scatter-xaxis',
                    clearable=False
                ),
            ])],
            style=style),

        html.Div([
            html.Label(["Behavior to show and correlate"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    behavior_initial,
                    id='behavior-scatter-yaxis',
                    clearable=False
                ),
            ])],
            style=style),

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
            style=style),

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
            style=style)

        ], style={'display': 'flex', 'padding': '10px 5px'}
    )


def build_regression_menu() -> html.Div:
    return html.Div([
        html.Label(['Style of regression line']),
        dcc.RadioItems(
            ['Overall regression', 'Rectified regression'],
            'Overall regression',
            id='regression-type',
            labelStyle={'display': 'inline-block', 'width': '33%'}
        )
        ]
    )


if __name__ == "__main__":
    main()
