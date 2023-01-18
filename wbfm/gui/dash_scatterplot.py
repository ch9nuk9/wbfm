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

    # Clicking callbacks
    @app.callback(
        Output('neuron-trace', 'figure'),
        Input('correlation-scatterplot', 'clickData'))
    def _update_trace(clickData):
        neuron_name = clickData["points"][0]["customdata"][0]
        _fig = make_trace_plot(df_behavior_and_traces, neuron_name)
        _fig.update_layout(height=800, margin={'l': 40, 'b': 40, 't': 30, 'r': 0})
        return _fig

    @app.callback(
        Output('trace-behavior-scatterplot', 'figure'),
        Input('correlation-scatterplot', 'clickData'),
        Input('behavior-scatter-yaxis', 'value')
    )
    def _update_second_scatter(clickData, behavior_name):
        neuron_name = clickData["points"][0]["customdata"][0]
        _fig = px.scatter(df_behavior_and_traces, x=behavior_name, y=neuron_name,
                          title=f"{neuron_name} and {behavior_name}",
                          trendline='ols')
        # Half as tall
        _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
        return _fig

    if __name__ == '__main__':
        app.run_server(debug=True)


def build_plots_html() -> html.Div:

    # Second trace plot, which is actually initialized through a clickData field on the scatterplot
    initial_clickData = {'points': [{'customdata': ['neuron_001']}]}

    left_side = html.Div([
            dcc.Graph(id='correlation-scatterplot', clickData=initial_clickData),
            dcc.Graph(id='trace-behavior-scatterplot')
    ], style={'width': '49%', 'display': 'inline-block'})

    right_side = html.Div([
            dcc.Graph(
                id='neuron-trace'
            )], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}
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

    return html.Div([
        html.Div([
            dcc.Dropdown(
                correlation_names_no_dummy,
                y_initial,
                id='scatter-yaxis',
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                correlation_names,
                x_initial,
                id='scatter-xaxis'
            ),
        ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                behavior_names,
                behavior_initial,
                id='behavior-scatter-yaxis',
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),

    ], style={'padding': '10px 5px'}
    )


def make_trace_plot(df_traces, neuron_name):
    fig_trace = px.line(df_traces, x=df_traces.index, y=neuron_name,
                        title=neuron_name)
    fig_trace.update_traces(mode='lines')
    return fig_trace


if __name__ == "__main__":
    main()
