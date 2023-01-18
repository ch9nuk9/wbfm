from pathlib import Path
from typing import List

from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd

from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df

DATA_FOLDER = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/tmp_data"


def main():

    app = Dash(__name__)
    for file in Path(DATA_FOLDER).iterdir():
        if 'df_traces' in file.name:
            df_traces = pd.read_hdf(file)
        elif 'df_correlation' in file.name:
            df_correlation = pd.read_hdf(file)
        elif 'df_behavior' in file.name:
            df_behavior = pd.read_hdf(file)

    df_behavior_and_traces = pd.concat([df_behavior, df_traces], axis=1)

    # Define layout
    app.layout = html.Div([
        build_dropdowns(df_correlation, df_behavior),
        build_regression_menu(),
        build_plots_html()
        ]
    )

    # Main scatter plot changing callback (change axes)
    @app.callback(
        Output('correlation-scatterplot', 'figure'),
        Input('scatter-xaxis', 'value'),
        Input('scatter-yaxis', 'value')
    )
    def update_scatter_plot(x_name, y_name):
        # Top scatter plot
        _fig = px.scatter(df_correlation, x=x_name, y=y_name, hover_name="neuron_name",
                          custom_data=["neuron_name"],
                          marginal_y='histogram',
                          title="Interactive Scatterplot")
        # Half as tall
        _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
        return _fig

    # Neuron selection updates
    @app.callback(
        Output('neuron-trace', 'figure'),
        Input('correlation-scatterplot', 'clickData'))
    def _update_neuron_trace(clickData):
        neuron_name = clickData["points"][0]["customdata"][0]
        _fig = make_neuron_trace_plot(df_behavior_and_traces, neuron_name)
        _fig.update_layout(height=325, margin={'l': 40, 'b': 40, 't': 30, 'r': 0})
        return _fig

    @app.callback(
        Output('trace-and-behavior-scatterplot', 'figure'),
        Input('correlation-scatterplot', 'clickData'),
        Input('behavior-scatter-yaxis', 'value'),
        Input('regression-type', 'value')
    )
    def _update_behavior_scatter(clickData, behavior_name, regression_type):
        if regression_type == 'Rectified regression':
            opt = {'color': 'reversal'}
        else:
            opt = {}
        neuron_name = clickData["points"][0]["customdata"][0]
        _fig = px.scatter(df_behavior_and_traces, x=behavior_name, y=neuron_name,
                          title=f"Behavior-neuron scatterplot",
                          trendline='ols', **opt)
        # Half as tall
        _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
        return _fig

    # Behavior updates
    @app.callback(
        Output('behavior-trace', 'figure'),
        Input('behavior-scatter-yaxis', 'value')
    )
    def _update_behavior_trace(behavior_name):
        _fig = px.line(df_behavior_and_traces, x=df_behavior_and_traces.index, y=behavior_name,
                       title=f"Trace of {behavior_name}")
        _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
        return _fig

    if __name__ == '__main__':
        app.run_server(debug=True)


def build_plots_html() -> html.Div:
    # Second trace plot, which is actually initialized through a clickData field on the scatterplot
    initial_clickData = {'points': [{'customdata': ['neuron_001']}]}

    left_side = html.Div([
            dcc.Graph(id='correlation-scatterplot', clickData=initial_clickData),
            dcc.Graph(id='trace-and-behavior-scatterplot')
    ], style={'width': '49%', 'display': 'inline-block'})

    right_side = html.Div([
            dcc.Graph(id='neuron-trace'),
            dcc.Graph(id='behavior-trace')],
        style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}
    )

    return html.Div([left_side, right_side])


def build_dropdowns(df_correlation, df_behavior) -> html.Div:
    correlation_names = get_names_from_df(df_correlation)
    correlation_names.remove('neuron_name')
    correlation_names_no_dummy = correlation_names.copy()
    correlation_names_no_dummy.remove('dummy')

    behavior_names = get_names_from_df(df_behavior)

    y_initial = 'coefficient'
    x_initial = 'dummy'
    behavior_initial = 'signed_speed'

    return html.Div(children=[
        html.Div([
            html.Label(['Scatter y axis'], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    correlation_names_no_dummy,
                    y_initial,
                    id='scatter-yaxis',
                ),
            ])],
            style={'width': '33%'}),

        html.Div([
            html.Label(['Scatter x axis'], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    correlation_names,
                    x_initial,
                    id='scatter-xaxis'
                ),
            ])],
            style={'width': '33%'}),

        html.Div([
            html.Label(["Behavior to show and correlate"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    behavior_initial,
                    id='behavior-scatter-yaxis',
                ),
            ])],
            style={'width': '33%'})

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


def make_neuron_trace_plot(df_traces, neuron_name):
    fig_trace = px.line(df_traces, x=df_traces.index, y=neuron_name, title=f"Trace for {neuron_name}")
    return fig_trace


if __name__ == "__main__":
    main()
