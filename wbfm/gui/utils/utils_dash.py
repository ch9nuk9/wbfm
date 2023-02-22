import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


def dashboard_from_two_dataframes(df: pd.DataFrame, df_summary: pd.DataFrame):
    """
    Create a dashboard from two dataframes. The first dataframe is used to create a line plot, and the second dataframe
    is used to create a scatter plot. The scatter plot has a clickData output, which is used to update the line plot.

    The index of the summary dataframe is the identifier of each point, and must correspond to a column in the original
    dataframe (df). The summary dataframe must have at least two columns, which will be used to create the scatter plot.

    Parameters
    ----------
    df
    df_summary

    Returns
    -------

    """
    app = dash.Dash(__name__)

    # Dash Bootstrap Components
    # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/

    # Get the column names of the summary dataframe
    column_names = df_summary.columns

    # Create a dropdown menu to choose each column of the summary dataframe
    dropdown_menu_x = dcc.Dropdown(
        id="dropdown_x",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[0],
        clearable=False
    )
    dropdown_menu_y = dcc.Dropdown(
        id="dropdown_y",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[1],
        clearable=False
    )

    dropdown_style = {'display': 'inline-block', 'width': '50%'}
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
        ]),
        html.Div([
            html.Div([
                dropdown_menu_x
            ], style=dropdown_style),
            html.Div([
                dropdown_menu_y
            ], style=dropdown_style),
        ]),

        html.Div([
            dcc.Graph(id="scatter", figure=px.scatter(df_summary, x="std", y="avg", hover_data=["stock"],
                                                      title="Click on a point to update the line plot"))
        ], style=row_style),
        html.Div([
            dcc.Graph(id="line", figure=px.line(df, x="date", y="GOOG", title="GOOG"))
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
        stock = clickData["points"][0]["customdata"][0]
        fig = px.line(df, x="date", y=stock, title=stock)
        return fig

    # Create a callback for updating the scatterplot using both dropdown menus
    # The scatter plot must have custom data, which is the name of the stock
    @app.callback(
        Output("scatter", "figure"),
        Input("dropdown_x", "value"),
        Input("dropdown_y", "value")
    )
    def update_scatter(x, y):
        fig = px.scatter(df_summary, x=x, y=y, hover_data=["stock"], custom_data=["stock"],
                         title="Click on a point to update the line plot")
        return fig

    return app


if __name__ == "__main__":
    # Create a test dataframe
    df = px.data.stocks()

    # Create a dataframe with the average and std of each stock as columns
    df_std = df.std()
    df_avg = df.mean()
    df_summary = pd.DataFrame({"std": df_std, "avg": df_avg})

    # Create a new column with the name of the stock
    df_summary["stock"] = df_summary.index

    app = dashboard_from_two_dataframes(df, df_summary)
    app.run_server(debug=True)
