import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Create a test dataframe
df = px.data.stocks()
# print(df)

# Create a dataframe with the average and std of each stock as columns
df_std = df.std()
df_avg = df.mean()
df_summary = pd.DataFrame({"std": df_std, "avg": df_avg})

# Create a new column with the name of the stock
df_summary["stock"] = df_summary.index

# Create a scatter plot showing the summary dataframe
# fig = px.scatter(df_summary, x="std", y="avg", hover_data=["stock"])
# fig.show()

# Plotly subplots
# https://plotly.com/python/subplots/
# fig = make_subplots(rows=1, cols=2)
#
# fig.add_trace(
#     px.scatter(df_summary, x="std", y="avg", hover_data=["stock"]).data[0],
#     row=1, col=1
# )
#
# fig.add_trace(
#     px.line(df, x="date", y="GOOG").data[0],
#     row=1, col=2
# )
#
# fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
# fig.show()

# Make a dash version of the plotly subplots above
# https://dash.plotly.com/layout
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Make a dash version of the plotly subplots above, with the scatterplot on the left and the line plot on the right
# The default layout is a column, so we need to make a row with two columns
# The scatter plot should have a clickData output, which will be used to update the line plot
# The scatter plot needs a default value of clickData, which will be used to update the line plot

# Dash Bootstrap Components
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/

# Create the layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Graph(id="scatter", figure=px.scatter(df_summary, x="std", y="avg", hover_data=["stock"]))
            ])),

        dbc.Col(
            html.Div([
                dcc.Graph(id="line", figure=px.line(df, x="date", y="GOOG"))
            ]))
        ]
    )
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
    # Debug:
    # print(clickData)
    stock = clickData["points"][0]["customdata"][0]
    fig = px.line(df, x="date", y=stock)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

# Write a julia version of the dash app above
#
#
# using Dash, DashBootstrapComponents, PlotlyJS
#
# app = dash(external_stylesheets=[dbc_themes.BOOTSTRAP])
#
# # Create the layout
# app.layout = html_div([
#     dbc_row([
#         dbc_col(
#             html_div([
#                 dcc_graph(id="scatter", figure=px_scatter(df_summary, x="std", y="avg", hover_data=["stock"]))
#             ])),
#
#         dbc_col(
#             html_div([
#                 dcc_graph(id="line", figure=px_line(df, x="date", y="GOOG"))
#             ]))
#         ]
#     )
# ])
#
#
# # Create a callback for clicking on the scatterplot. The callback should update the line plot
# @app.callback(
#     Output("line", "figure"),
#     Input("scatter", "clickData")
# )
# function update_line(clickData)
#     # Do not update if the clickData is None
#     if clickData is nothing
#         return dash_no_update
#     # Debug:
#     # print(clickData)
#     stock = clickData["points"][0]["customdata"][0]
#     fig = px_line(df, x="date", y=stock)
#     return fig
# end
#
#
# if __name__ == "__main__":
#     app.run_server(debug=true)



