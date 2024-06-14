import plotly.graph_objects as go
import plotly.express as px


def plotly_boxplot_colored_boxes(df, color_list):
    """
    Plotly can't color individual boxes using plotly express, so they have to be made one by one

    See https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

    Parameters
    ----------
    df
    color_list

    Returns
    -------

    """

    fig = go.Figure()

    columns = df.columns

    for i, (column, color) in enumerate(zip(columns, color_list)):
        fig.add_trace(go.Box(name=column, y=df[column], marker_color=color))

    fig.update_layout(showlegend=False)
    # Make one grid line at 0, and make it black
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


def add_trendline_annotation(fig):
    """
    Given a scatter plot with a trendline added, add an annotation with the slope, R² and p-value of the trendline

    Parameters
    ----------
    fig
    df

    Returns
    -------

    """
    # Extract trendline results
    results = px.get_trendline_results(fig)
    trendline = results.iloc[0].px_fit_results
    slope = trendline.params[1]
    r2 = trendline.rsquared
    pvalue = trendline.pvalues[1]

    # Get a reasonable position (top right) for the annotation
    x = fig.layout.xaxis.range[1]
    y = fig.layout.yaxis.range[1]

    # Add the annotation
    annotation_text = f'Slope: {slope:.2f}<br>R²: {r2:.2f}<br>p-value: {pvalue:.3f}'
    fig.add_annotation(
        x=x, y=y,
        text=annotation_text,
        showarrow=False,
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        bgcolor='white',
        opacity=0.8
    )

    return fig
