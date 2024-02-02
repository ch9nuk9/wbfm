import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from wbfm.utils.tracklets.postprocess_tracking import OutlierRemoval
from wbfm.utils.utils_cache import cache_to_disk_class


def paper_trace_settings():
    """
    The settings used in the paper.

    Returns
    -------

    """
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.75,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=True,
               channel_mode='dr_over_r_50',
               use_physical_time=True,
               rename_neurons_using_manual_ids=True,
               always_keep_manual_ids=True,
               manual_id_confidence_threshold=0)
    return opt


def plotly_paper_color_discrete_map():
    """
    To be used with the color_discrete_map argument of plotly.express functions

    Parameters
    ----------
    plotly_not_matplotlib

    Returns
    -------

    """
    base_cmap = px.colors.qualitative.D3
    cmap_dict = {'gcamp': base_cmap[0], 'wbfm': base_cmap[0], 'immob': base_cmap[1], 'gfp': base_cmap[2],
                 'global': base_cmap[3], 'residual': base_cmap[4]}
    # Add alternative names
    for k, v in data_type_name_mapping().items():
        cmap_dict[v] = cmap_dict[k]
    return cmap_dict


def data_type_name_mapping():
    return {'wbfm': 'Freely Moving (GCaMP)',
            'gcamp': 'Freely Moving (GCaMP)',
            'immob': 'Immobilized (GCaMP)',
            'gfp': 'Freely Moving (GFP)'}


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
    # Note: changes this globally
    # plt.rcParams["font.family"] = "arial"

    matplotlib_opt = dict(figsize=(column_width_inches*width_factor,
                                   column_height_inches*height_factor), dpi=dpi)
    matplotlib_font_opt = dict(fontsize=font_size_points)
    plotly_opt = dict(width=round(column_width_pixels*width_factor),
                      height=round(column_height_pixels*height_factor))
    # See: https://stackoverflow.com/questions/67844335/what-is-the-default-font-in-python-plotly
    plotly_font_opt = dict(font=dict(size=font_size_pixels, color='black'), font_family="arial")

    opt = dict(matplotlib_opt=matplotlib_opt, plotly_opt=plotly_opt,
               matplotlib_font_opt=matplotlib_font_opt, plotly_font_opt=plotly_font_opt)
    return opt


def apply_figure_settings(fig=None, width_factor=1, height_factor=1, plotly_not_matplotlib=True):
    """
    Apply settings for the paper, per figure. Note that this does not change the size settings, only font sizes and
    background colors (transparent).

    Parameters
    ----------
    fig - Figure to modify. If None, will use plt.gcf(), which assumes that the figure is the current matplotlib figure
    width_factor - Fraction of an A4 page to use (width)
    height_factor - Fraction of an A4 page to use (height)
    plotly_not_matplotlib - If True, will modify the figure using plotly syntax. Otherwise, will use matplotlib syntax

    Returns
    -------

    """
    if fig is None:
        if not plotly_not_matplotlib:
            fig = plt.gcf()
        else:
            raise NotImplementedError("Only matplotlib is supported if the figure is not directly passed for now")
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
        # Add black lines on edges of plot (only left and bottom
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
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


def behavior_name_mapping():
    name_mapping = dict(
        signed_middle_body_speed='Speed',
        dorsal_only_head_curvature='Dorsal head curvature',
        ventral_only_head_curvature='Ventral head curvature',
        dorsal_only_body_curvature='Dorsal curvature',
        ventral_only_body_curvature='Ventral curvature',
        FWD='Forward crawling',
        REV='Backward crawling',
        VENTRAL_TURN='Ventral turning',
        DORSAL_TURN='Dorsal turning',
    )
    return name_mapping


class PaperDataCache:
    """
    Class for caching data generated by the project data, and to be used in the figures of the paper.

    """

    def __init__(self, project_data):

        from wbfm.utils.projects.finished_project_data import ProjectData
        self.project_data: ProjectData = project_data

    @cache_to_disk_class('invalid_indices_cache_fname',
                         func_save_to_disk=np.save,
                         func_load_from_disk=np.load)
    def calc_indices_to_remove_using_ppca(self):
        names = self.project_data.neuron_names
        coords = ['z', 'x', 'y']
        all_zxy = self.project_data.red_traces.loc[:, (slice(None), coords)].copy()
        z_to_xy_ratio = self.project_data.physical_unit_conversion.z_to_xy_ratio
        all_zxy.loc[:, (slice(None), 'z')] = z_to_xy_ratio * all_zxy.loc[:, (slice(None), 'z')]
        outlier_remover = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces=None, names=names, verbose=0)
        outlier_remover.iteratively_remove_outliers_using_ppca(max_iter=8)
        to_remove = outlier_remover.total_matrix_to_remove
        return to_remove

    def invalid_indices_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'invalid_indices.npy')

    @cache_to_disk_class('paper_traces_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces.h5')

    @cache_to_disk_class('paper_traces_cache_fname_red',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_red(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['channel_mode'] = 'red'
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname_red(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_red.h5')

    @cache_to_disk_class('paper_traces_cache_fname_green',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_green(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['channel_mode'] = 'green'
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname_green(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_green.h5')

    @cache_to_disk_class('paper_traces_residual_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_residual(self):
        """
        Like calc_paper_traces but adds the residual mode.
        """
        opt = paper_trace_settings()
        opt['residual_mode'] = 'pca'
        opt['interpolate_nan'] = True
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces (residual) for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_residual_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_residual.h5')

    @cache_to_disk_class('paper_traces_global_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_global(self):
        """
        Like calc_paper_traces but for the global mode.
        """
        opt = paper_trace_settings()
        opt['residual_mode'] = 'pca_global'
        opt['interpolate_nan'] = True
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces (global) for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_global_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_global.h5')

    @property
    def cache_dir(self):
        fname = os.path.join(self.project_data.project_dir, '.cache')
        if not os.path.exists(fname):
            try:
                os.makedirs(fname)
            except PermissionError:
                print(f"Could not create cache directory {fname}")
                fname = None
        return fname

    def clear_disk_cache(self, delete_traces=True, delete_invalid_indices=True,
                         dry_run=False, verbose=1):
        """
        Deletes all cached files generated using the cache_to_disk_class decorator

        Returns
        -------

        """
        possible_fnames = []
        if delete_traces:
            possible_fnames.append(self.paper_traces_cache_fname())
            possible_fnames.append(self.paper_traces_residual_cache_fname())
            possible_fnames.append(self.paper_traces_global_cache_fname())
        if delete_invalid_indices:
            possible_fnames.append(self.invalid_indices_cache_fname())
        for fname in possible_fnames:
            if os.path.exists(fname):
                if verbose >= 1:
                    print(f"Deleting {fname}")
                if not dry_run:
                    os.remove(fname)


def neurons_with_confident_ids():
    neuron_names = ['AVAL', 'AVAR', 'BAGL', 'BAGR', 'RIMR', 'RIML', 'AVEL', 'AVER', 'RIVR', 'RIVL', 'SMDVL', 'SMDVR',
                    'ALA', 'RIS',
                    'VB02', 'RIBL', 'RIBR', 'RMEL', 'RMER', 'RMED', 'RMEV', 'RID', 'AVBL', 'AVBR']
    return neuron_names
