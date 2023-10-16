
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
