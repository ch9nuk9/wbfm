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
column_width_inches = 5.4  # From elsevier template
column_width_pixels = column_width_inches * dpi
column_height_inches = 7.78  # From elsevier template
column_height_pixels = column_height_inches * dpi
pixels_per_point = dpi / 72.0
font_size_points = 18
font_size_pixels = font_size_points * pixels_per_point


def paper_figure_page_settings(height_factor=1, width_factor=1):
    """Settings for a full column width, full height. Will be multiplied later"""
    matplotlib_opt = dict(figsize=(column_width_inches*width_factor,
                                   column_height_inches*height_factor), dpi=dpi)
    matplotlib_font_opt = dict(fontsize=font_size_points)
    plotly_opt = dict(width=column_width_pixels*width_factor,
                      height=column_height_pixels*height_factor)
    plotly_font_opt = dict(font=dict(size=font_size_pixels))

    opt = dict(matplotlib_opt=matplotlib_opt, plotly_opt=plotly_opt,
               matplotlib_font_opt=matplotlib_font_opt, plotly_font_opt=plotly_font_opt)
    return opt


# def paper_figure_1_settings():
#     base_font_size = 18
#     pixel_width = column_width_pixels
#     pixel_height = 400
#
#     return dict(font_size_points=font_size_points,
#                 base_width=pixel_width,
#                 base_height=pixel_height)
#
#
# def paper_figure_2_settings():
#     pixel_width = column_width_pixels / 3.0
#     pixel_height = 400
#
#     return dict(font_size_points=font_size_points,
#                 base_width=pixel_width,
#                 base_height=pixel_height)
#
#
# def paper_figure_3_settings():
#     base_font_size = 18
#     pixel_width = 96*5.4  # Column width
#     pixel_height = pixel_width / 1.618  # Golden ratio
#
#     opt = dict(base_font_size=base_font_size)
#     opt['matplotlib_opt'] = dict(figsize=(5.4, 2), dpi=96)
#     opt['plotly_opt'] = dict(width=pixel_width, height=pixel_height)
#     return opt


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
        # Update font size
        fig.update_layout(**font_dict, title=font_dict)
        # Transparent background
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # Remove background grid lines
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
    else:
        font_dict = figure_opt['matplotlib_font_opt']
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

