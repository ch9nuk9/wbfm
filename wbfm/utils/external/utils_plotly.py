import numpy as np
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

    # Get a reasonable position (top right) for the annotation using the x and y max data points
    x = np.nanmin(fig.data[0].x)
    y = np.nanmax(fig.data[0].y)

    # Add the annotation
    annotation_text = f'Slope: {slope:.2f}<br>R²: {r2:.2f}<br>p-value: {pvalue:.2e}'
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


def plotly_plot_mean_and_shading(df, x, y, color, line_name='Mean', add_individual_lines=False,
                                 cmap=None, x_intersection_annotation=None, annotation_kwargs=None,
                                 annotation_position='left', fig=None):
    """
    Plot the mean of a y column for each x value, and shade the standard deviation

    Note that this requires the identical x values for each group

    Parameters
    ----------
    df
    x
    y

    Returns
    -------

    """
    if annotation_kwargs is None:
        annotation_kwargs = dict()

    # Calculate mean and std dev for each x value
    grouped = df.groupby(x)
    mean_y = grouped[y].mean()
    std_y = grouped[y].std()

    if fig is None:
        fig = go.Figure()

    if add_individual_lines:
        for group in df[color].unique():
            df_subset = df[df[color] == group]
            fig.add_trace(go.Scatter(
                x=df_subset[x], y=df_subset[y], mode='lines', name=f'Group {group}', line=dict(width=1)
            ))

    # Add the mean line
    opt = dict()
    if cmap is not None:
        opt['line'] = dict(color=cmap[line_name])
    fig.add_trace(go.Scatter(
        x=mean_y.index, y=mean_y, mode='lines', name=line_name, **opt
    ))

    # Shade the standard deviation area
    opt = dict()
    if cmap is not None:
        hex_color = cmap[line_name]
        # Convert hex to rgba
        opt['fillcolor'] = f"rgba{tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + (0.2,)}"
    else:
        opt['fillcolor'] = 'rgba(0,100,80,0.2)'

    fig.add_trace(go.Scatter(
        x=np.concatenate([mean_y.index, mean_y.index[::-1]]),
        y=np.concatenate([mean_y + std_y, (mean_y - std_y)[::-1]]),
        fill='toself',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        **opt
    ))

    if x_intersection_annotation is not None:
        y_value_at_x = mean_y.loc[x_intersection_annotation]
        # Add vertical line at x=x_intersection_annotation
        fig.add_shape(type="line",
                      x0=x_intersection_annotation, y0=mean_y.min(), x1=x_intersection_annotation, y1=mean_y.max(),
                      line=dict(color="Black", width=1, dash="dash"),
                      )

        # Add horizontal line at the intersection with mean_y
        fig.add_shape(type="line",
                      x0=mean_y.index.min(), y0=y_value_at_x, x1=mean_y.index.max(), y1=y_value_at_x,
                      line=dict(color="Black", width=1, dash="dash"),
                      )

        # Add text annotation at the intersection point
        x = 0.5*x_intersection_annotation if annotation_position == 'left' else 1.5*x_intersection_annotation
        fig.add_annotation(
            x=x, y=1.1*y_value_at_x,
            text=f"y={y_value_at_x:.2f}",
            showarrow=False,
            **annotation_kwargs
            # showarrow=True,
            # arrowhead=2,
            # ax=-40, ay=-10  # Position the text relative to the point
        )

    return fig

