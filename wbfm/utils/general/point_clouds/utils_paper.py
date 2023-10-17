
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


def paper_figure_1_settings():
    base_font_size = 18
    base_width = 1.5 * 1000
    base_height = 400

    return dict(base_font_size=base_font_size,
                base_width=base_width,
                base_height=base_height)


def paper_figure_2_settings():
    base_font_size = 18
    base_width = 1.5 * 1000 / 3.0
    base_height = 400

    return dict(base_font_size=base_font_size,
                base_width=base_width,
                base_height=base_height)


def apply_figure_settings(fig, i_figure):
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
    if i_figure == 1:
        figure_opt = paper_figure_1_settings()
    elif i_figure == 2:
        figure_opt = paper_figure_2_settings()
    else:
        raise ValueError('Unknown figure number: {}'.format(i_figure))
    base_font_size = figure_opt['base_font_size']
    base_width = figure_opt['base_width']
    base_height = figure_opt['base_height']

    # Update font size
    fig.update_layout(font=dict(size=base_font_size),
                      title=dict(font=dict(size=base_font_size+2)))
    # Transparent background and remove lines
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")