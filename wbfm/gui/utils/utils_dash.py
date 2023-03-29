import os
import pickle
from typing import Dict

import dash
import numpy as np
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go


def dashboard_from_two_dataframes(df_summary: pd.DataFrame, raw_dfs: Dict[str, pd.DataFrame], is_jupyter=False,
                                  x_default=None, y_default=None, color_default=None):
    """
    Create a dashboard from two dataframes. The first dataframe is used to create a line plot, and the second dataframe
    is used to create a scatter plot. The scatter plot has a clickData output, which is used to update the line plot.

    The index of the summary dataframe is the identifier of each point, and must correspond to a column in the original
    dataframe (raw_dfs). The summary dataframe must have at least two columns, which will be used to create the scatter plot.

    In other words:
        df_summary.columns = variables that can be used in the scatter plot
        df_summary.index = identifier of each point in the scatter plot
            (e.g. neuron name, or neuron name concatenated with dataset name)

        raw_dfs[key].columns = df_summary.index
        raw_dfs[key].index = time-like variable (e.g. time, frame number, etc.)

    In addition,
        df_summary must have a column called 'index' which is the column name of each raw_dfs dataframe
        i.e. df_summary['index'] = df_summary.index = raw_dfs[key].columns

    Examples:
        python interactive_two_dataframe_gui.py -p '/home/charles/Current_work/presentations/Feb_2023/gui_volcano_plot_kymograph/' -x 'body_segment_argmax' -y 'corr_max' -c 'genotype'

        python interactive_two_dataframe_gui.py -p '/home/charles/Current_work/presentations/Feb_2023/gui_volcano_plot_rev_triggered/' --allow_public_access True -x 'effect size' -y '-log(p value)' -c 'genotype'

        python interactive_two_dataframe_gui.py -p '/home/charles/Current_work/presentations/Feb_2023/gui_behavior_and_pc0' -x 'datatype' -y 'R2 scores' --allow_public_access True --port 8065

    Parameters
    ----------
    raw_dfs
    df_summary

    Returns
    -------

    """
    if is_jupyter:
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)
    else:
        app = dash.Dash(__name__)

    # Check for necessary columns in df_summary
    if 'index' not in df_summary.columns:
        df_summary['index'] = df_summary.index
    elif not df_summary['index'].equals(df_summary.index):
        df_summary.index = df_summary['index']

    # Get the column names of the summary dataframe
    column_names = df_summary.columns
    column_names_with_none = ['None'] + list(column_names)

    keys = list(raw_dfs.keys())
    initial_click_point = raw_dfs[keys[0]].columns[0]
    print(f"Initial click point: {initial_click_point}")
    initial_clickData = {'points': [{'customdata': [initial_click_point]}]}

    # Create a dropdown menu to choose each column of the summary dataframe
    dropdown_menu_x = dcc.Dropdown(
        id="dropdown_x",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[0] if x_default is None else x_default,
        clearable=False
    )
    dropdown_menu_y = dcc.Dropdown(
        id="dropdown_y",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[1] if y_default is None else y_default,
        clearable=False
    )
    dropdown_menu_color = dcc.Dropdown(
        id="dropdown_color",
        options=[{"label": i, "value": i} for i in column_names_with_none],
        value=color_default,
        clearable=False
    )

    dropdown_style = {'display': 'inline-block', 'width': '33%'}
    row_style = {'display': 'inline-block', 'width': '100%'}
    # Create the layout
    app.layout = html.Div([
        # Create two rows, with a label on top of a dropdown menu
        html.Div([
            html.Div([
                html.Label("X axis (scatterplot)"),
            ], style=dropdown_style),
            html.Div([
                html.Label("Y axis (scatterplot)"),
            ], style=dropdown_style),
            html.Div([
                html.Label("Color splitting (scatterplot)"),
            ], style=dropdown_style),
        ]),
        html.Div([
            html.Div([
                dropdown_menu_x
            ], style=dropdown_style),
            html.Div([
                dropdown_menu_y
            ], style=dropdown_style),
            html.Div([
                dropdown_menu_color
            ], style=dropdown_style),
        ]),

        html.Div([
            dcc.Graph(id="scatter", clickData=initial_clickData),
        ], style=row_style),
        html.Div([
            dcc.Graph(id="line")
        ], style=row_style)
    ])

    # Create a callback for clicking on the scatterplot. The callback should update the line plot
    @app.callback(
        Output("line", "figure"),
        Input("scatter", "clickData")
    )
    def update_line(clickData):
        # Do not update if the clickData is None
        if clickData is None:
            return dash.no_update
        try:
            click_name = clickData["points"][0]["customdata"][0]
        except TypeError:
            return dash.no_update

        try:
            fig = line_plot_from_dict_of_dataframes(raw_dfs, y=click_name)
        except ValueError:
            # Sometimes plotly fails the first time, so try again
            # https://stackoverflow.com/questions/74367104/dashboard-plotly-valueerror-invalid-value
            fig = line_plot_from_dict_of_dataframes(raw_dfs, y=click_name)

        return fig

    # Create a callback for updating the scatterplot using both dropdown menus
    # The scatter plot must have custom data, which is the name of the stock
    @app.callback(
        Output("scatter", "figure"),
        Input("dropdown_x", "value"),
        Input("dropdown_y", "value"),
        Input("dropdown_color", "value"),
        Input("scatter", "clickData"),
        State("scatter", "figure")
    )
    def update_scatter(x, y, color, clickData, fig):
        id_input_changed = dash.ctx.triggered_id
        selected_row = clickData["points"][0]["customdata"][0]
        # Regenerate the full plot if the dropdown menu is changed, but not if the scatterplot is clicked
        # Also regenerate on the first load
        if id_input_changed == "scatter" and fig is not None:
            # First, check in which dictionary the selected point exists
            # Then modify that one, while resetting the others
            all_customdata = [fig['data'][i]['customdata'] for i in range(len(fig['data']))]
            for i, customdata in enumerate(all_customdata):
                num_pts = len(customdata)
                new_sz = np.ones(num_pts)
                if [selected_row] in customdata:
                    i_selected_row = customdata.index([selected_row])
                    new_sz[i_selected_row] = 5
                    fig['data'][i]['marker']['size'] = new_sz
                else:
                    fig['data'][i]['marker']['size'] = new_sz
        else:
            fig = _build_scatter_plot(df_summary, x, y, selected_row, color=color)

        return fig

    return app


def _build_scatter_plot(df_summary, x, y, selected_row, **kwargs):
    df_summary['selected'] = 1
    df_summary.loc[selected_row, 'selected'] = 5
    fig = px.scatter(df_summary, x=x, y=y, hover_data=["index"], custom_data=["index"],
                     size='selected',
                     title="Click on a point to update the line plot",
                     marginal_y="violin",
                     **kwargs)
    fig.update_layout(font=dict(size=18))
    return fig


def line_plot_from_dict_of_dataframes(dict_of_dfs: dict, y: str):
    # Create a single figure from a list of dataframes
    # dict_of_dfs is a nested dictionary. The first key is the name of the trace type (e.g. 'speed')
    #    The second key is the name of the trace inlcuding dataset and neuron
    #    e.g. 'ZIM2165_Gcamp7b_worm1-2022_11_28'

    # Build a temporary dataframe to plot
    df = pd.DataFrame({k: df_[y].sort_index() for k, df_ in dict_of_dfs.items()})
    fig = px.line(df)

    fig.update_layout(xaxis_title="Time", yaxis_title="Amplitude", font=dict(size=18))
    return fig

#
# if __name__ == "__main__":
#     # Create a test dataframe
#     df = px.data.stocks()
#     raw_dfs = {'test': df}
#
#     # Create a dataframe with the average and std of each stock as columns
#     df_std = df.std()
#     df_avg = df.mean()
#     df_summary = pd.DataFrame({"std": df_std, "avg": df_avg})
#
#     # Create a new column with the name of the stock
#     df_summary["stock"] = df_summary.index
#
#     app = dashboard_from_two_dataframes(df_summary, raw_dfs)
#     app.run_server(debug=True)


def save_folder_for_two_dataframe_dashboard(output_folder, df_summary: pd.DataFrame, raw_dfs: Dict[str, pd.DataFrame]):
    """
    See the docstring of dashboard_from_two_dataframes for more information

    Parameters
    ----------
    output_folder
    df_summary
    raw_dfs

    Returns
    -------

    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    df_summary.to_hdf(os.path.join(output_folder, 'df_summary.h5'), key='df_summary')
    with open(os.path.join(output_folder, 'raw_dfs.pickle'), 'wb') as handle:
        pickle.dump(raw_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
