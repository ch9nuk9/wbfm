import networkx as nx
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


def plotly_plot_mean_and_shading(df, x, y, color=None, line_name='Mean', add_individual_lines=False,
                                 cmap=None, x_intersection_annotation=None, annotation_kwargs=None,
                                 annotation_position='left', fig=None, **kwargs):
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

    if color is not None and len(df[color].unique()) > 1:
        # Assume we want to subset the dataframe by the color list
        fig = None
        annotation_position = 'top left'
        for group in df[color].unique():
            _df = df[df[color] == group]
            # Alternate annotation_position by default
            annotation_position = 'bottom right' if annotation_position == 'top left' else 'top left'

            fig = plotly_plot_mean_and_shading(_df, x, y, color=color, line_name=group,
                                               add_individual_lines=False, cmap=cmap, fig=fig,
                                               x_intersection_annotation=x_intersection_annotation,
                                               annotation_position=annotation_position)
        return fig

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
                x=df_subset[x], y=df_subset[y], mode='lines', name=f'Group {group}', line=dict(width=1), **kwargs
            ))

    # Add the mean line
    opt = dict()
    if cmap is not None:
        opt['line'] = dict(color=cmap[line_name])
    fig.add_trace(go.Scatter(
        x=mean_y.index, y=mean_y, mode='lines', name=str(line_name), **opt
    ))

    # Shade the standard deviation area
    opt = dict()
    if cmap is not None:
        opt['fillcolor'] = hex2rgba(cmap[line_name])
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
        x = 0.5*x_intersection_annotation if 'left' in annotation_position else 1.5*x_intersection_annotation
        y = 1.1*y_value_at_x if 'top' in annotation_position else 0.9*y_value_at_x
        fig.add_annotation(
            x=x, y=y,
            text=f"y={y_value_at_x:.2f}",
            showarrow=False,
            **annotation_kwargs
            # showarrow=True,
            # arrowhead=2,
            # ax=-40, ay=-10  # Position the text relative to the point
        )

    return fig


def hex2rgba(hex_color, alpha=0.2):
    fillcolor = f"rgba{tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)}"
    return fillcolor


def get_nonoverlapping_text_positions(x, y, text, fig, weight=100, k=None):
    positions = np.array(list(zip(x, y)))
    G = nx.Graph()

    # Add nodes
    fixed_nodes = []
    for i, pos in enumerate(positions):
        # if len(text[i]) == 0:
        #     # Skip empty text
        #     continue
        G.add_node(f"text_{i}", pos=pos)
        G.add_node(f"data_{i}", pos=pos)  # Will be fixed in place
        fixed_nodes.append(f"data_{i}")

    # Add edges based on distance (short distances mean a stronger repulsion)
    for i in range(len(positions)):
        G.add_edge(f"data_{i}", f"text_{i}", weight=weight)  # Try to keep the text near the data
        # for j in range(i + 1, len(positions)):
        #     dist = np.linalg.norm(positions[i] - positions[j])
        #     G.add_edge(f"text_{i}", f"text_{j}", weight=1.0 / (dist + 1e-4))
        # print(1.0 / (dist + 1e-4))

    # Apply force-directed layout
    new_positions = nx.spring_layout(G, pos=nx.get_node_attributes(G, 'pos'), fixed=fixed_nodes,
                                     weight='weight',  k=k,)

    # Update the plot with new text positions
    adjusted_text_positions = np.array([new_positions[f"text_{i}"] for i in range(len(positions))])

    # Create new scatter plot with adjusted text positions
    if fig is None:
        adjusted_scatter = go.Scatter(
            x=x, y=y,
            mode='markers',
        )
        fig = go.Figure(data=[adjusted_scatter])

    for i, (t, (x_new, y_new)) in enumerate(zip(text, adjusted_text_positions)):
        _x, _y = x[i], y[i]
        if len(t) == 0:
            continue
        fig.add_annotation(x=_x, y=_y, ax=x_new, ay=y_new,  # arrowhead=2,
                           text=t, xref="x", yref="y", axref="x", ayref="y", )

    return fig
