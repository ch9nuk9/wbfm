import matplotlib.pyplot as plt


def paper_trace_settings():
    """
    The settings used in the paper.

    Returns
    -------

    """
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=True,
               channel_mode='dr_over_r_50',
               use_physical_time=True,
               rename_neurons_using_manual_ids=True)
    return opt


# Basic settings based on the physical dimensions of the paper
dpi = 96
# column_width_inches = 6.5  # From 3p elsevier template
column_width_inches = 8.5  # Full a4 page
column_width_pixels = column_width_inches * dpi
# column_height_inches = 8.6  # From 3p elsevier template
column_height_inches = 11  # Full a4 page
column_height_pixels = column_height_inches * dpi
pixels_per_point = dpi / 72.0
font_size_points = 10  # I think the default is 10, but since I am doing a no-margin image I need to be a bit larger
font_size_pixels = font_size_points * pixels_per_point


def paper_figure_page_settings(height_factor=1, width_factor=1):
    """Settings for a full column width, full height. Will be multiplied later"""
    matplotlib_opt = dict(figsize=(column_width_inches*width_factor,
                                   column_height_inches*height_factor), dpi=dpi)
    matplotlib_font_opt = dict(fontsize=font_size_points)
    plotly_opt = dict(width=round(column_width_pixels*width_factor),
                      height=round(column_height_pixels*height_factor))
    # plotly_opt = dict(width=3840,
    #                   height=1600)
    plotly_font_opt = dict(font=dict(size=font_size_pixels))

    opt = dict(matplotlib_opt=matplotlib_opt, plotly_opt=plotly_opt,
               matplotlib_font_opt=matplotlib_font_opt, plotly_font_opt=plotly_font_opt)
    return opt


def apply_figure_settings(fig, width_factor=1, height_factor=1, plotly_not_matplotlib=True):
    """
    Apply settings for the paper, per figure. Note that this does not change the size settings, only font sizes and
    background colors (transparent).

    Parameters
    ----------
    fig
    i_figure

    Returns
    -------

    """
    figure_opt = paper_figure_page_settings(width_factor=width_factor, height_factor=height_factor)

    if plotly_not_matplotlib:
        font_dict = figure_opt['plotly_font_opt']
        size_dict = figure_opt['plotly_opt']
        # Update font size
        fig.update_layout(**font_dict, **size_dict, title=font_dict, autosize=False)
        # Transparent background
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # Remove background grid lines
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        # Remove margin
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    else:
        font_dict = figure_opt['matplotlib_font_opt']
        size_dict = figure_opt['matplotlib_opt']
        # Change size
        fig.set_size_inches(size_dict['figsize'])
        fig.set_dpi(size_dict['dpi'])

        # Get ax from figure
        ax = fig.axes[0]

        # Title font size
        title = ax.title
        title.set_fontsize(font_dict['fontsize'])

        # X-axis and Y-axis label font sizes
        xlabel = ax.xaxis.label
        ylabel = ax.yaxis.label
        xlabel.set_fontsize(font_dict['fontsize'])
        ylabel.set_fontsize(font_dict['fontsize'])

        # Tick label font sizes
        for tick in ax.get_xticklabels():
            tick.set_fontsize(font_dict['fontsize'])
        for tick in ax.get_yticklabels():
            tick.set_fontsize(font_dict['fontsize'])

        plt.tight_layout()
