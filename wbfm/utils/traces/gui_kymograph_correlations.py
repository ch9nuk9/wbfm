# Helper functions
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wbfm.gui.utils.utils_dash import save_folder_for_two_dataframe_dashboard
from wbfm.utils.external.utils_pandas import flatten_multiindex_columns
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding


def build_all_gui_dfs_multineuron_correlations(all_projects_gcamp: Dict[str, ProjectData],
                                               all_projects_gfp: Dict[str, ProjectData],
                                               output_folder: str = None,
                                               trace_options=None,
                                               posture_attribute='curvature',
                                               min_manual_id_confidence=2,
                                               **kwargs):
    """
    Builds all the dataframes needed for the GUI, including reverse and forward rectification.

    Parameters
    ----------
    all_projects_gcamp
    all_projects_gfp
    output_folder
    trace_options
    posture_attribute

    Returns
    -------

    """
    # Use same trace options for all plots
    if trace_options is None:
        trace_options = {}
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=False,  # This takes a long time!
               channel_mode='dr_over_r_50')
    opt.update(trace_options)

    # Dataframes for the summary table
    df_summary_gcamp, df_summary_gcamp_rev, df_summary_gcamp_fwd = build_all_summary_dfs(all_projects_gcamp, opt,
                                                                                         posture_attribute, **kwargs)
    df_summary_gcamp['genotype'] = 'gcamp'
    df_summary_gcamp_rev['genotype'] = 'gcamp'
    df_summary_gcamp_fwd['genotype'] = 'gcamp'

    df_summary_gfp, df_summary_gfp_rev, df_summary_gfp_fwd = build_all_summary_dfs(all_projects_gfp, opt,
                                                                                   posture_attribute, **kwargs)
    df_summary_gfp['genotype'] = 'gfp'
    df_summary_gfp_rev['genotype'] = 'gfp'
    df_summary_gfp_fwd['genotype'] = 'gfp'

    df_summary = pd.concat([df_summary_gcamp, df_summary_gfp])
    df_summary_rev = pd.concat([df_summary_gcamp_rev, df_summary_gfp_rev])
    df_summary_fwd = pd.concat([df_summary_gcamp_fwd, df_summary_gfp_fwd])

    # Dataframes for individual traces
    df_trace_gcamp = build_trace_dfs(all_projects_gcamp, opt)
    df_trace_gfp = build_trace_dfs(all_projects_gfp, opt)
    df_traces = pd.concat([df_trace_gcamp, df_trace_gfp], axis=1)
    df_traces = flatten_multiindex_columns(df_traces)

    # Dataframes for best segment (correlation)
    df_seg_gcamp = build_best_segment_dfs(all_projects_gcamp, df_summary_gcamp)
    df_seg_gfp = build_best_segment_dfs(all_projects_gfp, df_summary_gfp)
    df_all_segs = pd.concat([df_seg_gcamp, df_seg_gfp], axis=1)

    # Additional columns: quantiles
    all_dfs = [df_summary, df_summary_rev, df_summary_fwd]
    for df in all_dfs:
        add_quantile_columns(df)

    # Additional columns: manually id'ed neuron name
    all_projects = {**all_projects_gcamp, **all_projects_gfp}
    func = partial(get_manual_annotation_from_project, all_projects, min_confidence=min_manual_id_confidence)
    for df in all_dfs:
        df_index = list(df.index)
        df_new_col = build_new_column_from_function(df_index, func)
        df['manual_id'] = df_new_col

    # Save
    raw_dfs = {'best segment': df_all_segs, 'trace': df_traces}
    if output_folder is not None:
        save_folder_for_two_dataframe_dashboard(output_folder, df_summary, raw_dfs)
        output_folder_rev = output_folder + '_rev'
        save_folder_for_two_dataframe_dashboard(output_folder_rev, df_summary_rev, raw_dfs)
        output_folder_fwd = output_folder + '_fwd'
        save_folder_for_two_dataframe_dashboard(output_folder_fwd, df_summary_fwd, raw_dfs)

    return df_summary, df_summary_rev, df_summary_fwd, raw_dfs


def add_quantile_columns(df):
    df['var_brightness_quantile'] = np.argsort(df.var_brightness) / df.shape[0]
    df['median_brightness_quantile'] = np.argsort(df.median_brightness) / df.shape[0]

    return df


def build_all_summary_dfs(all_projects, opt, posture_attribute: str = "curvature", **kwargs):
    dataframes_to_load = opt['channel_mode']

    def _get_df_tmp(p, rectification_variable):
        encoder = NeuronToMultivariateEncoding(p, df_kwargs=opt.copy(),
                                               dataframes_to_load=[dataframes_to_load],
                                               posture_attribute=posture_attribute, **kwargs)
        df_tmp = encoder.calc_per_neuron_df(dataframes_to_load, rectification_variable)
        df_tmp['neuron_name'] = df_tmp.index
        df_tmp.index = df_tmp['dataset_name'].astype(str) + '_' + df_tmp['neuron_name']
        return df_tmp

    all_dfs = []
    all_dfs_rev = []
    all_dfs_fwd = []
    for k, p in tqdm(all_projects.items(), leave=False):
        all_dfs.append(_get_df_tmp(p, None))
        all_dfs_rev.append(_get_df_tmp(p, 'rev'))
        all_dfs_fwd.append(_get_df_tmp(p, 'fwd'))

    df_summary = pd.concat(all_dfs)
    df_summary['index'] = df_summary.index
    df_summary_rev = pd.concat(all_dfs_rev)
    df_summary_rev['index'] = df_summary_rev.index
    df_summary_fwd = pd.concat(all_dfs_fwd)
    df_summary_fwd['index'] = df_summary_fwd.index
    return df_summary, df_summary_rev, df_summary_fwd


def build_trace_dfs(all_projects, opt):
    all_dfs = {}
    for k, p in all_projects.items():
        df_tmp = p.calc_default_traces(**opt)
        all_dfs[k] = df_tmp

    df_summary = pd.concat(all_dfs, axis=1)
    return df_summary


def build_best_segment_dfs(all_projects, df_summary, posture_attribute='curvature'):
    df_dict = dict()
    for i, row in df_summary.iterrows():
        col_name = row['dataset_name'] + '_' + row['neuron_name']
        i_argmax = row['body_segment_argmax']
        p = all_projects[row['dataset_name']]

        kymo = getattr(p.worm_posture_class, posture_attribute)(fluorescence_fps=True)
        kymo.reset_index(inplace=True, drop=True)

        best_seg = kymo.iloc[:, i_argmax]

        df_dict[col_name] = best_seg

    df_seg = pd.DataFrame(df_dict)
    return df_seg


def build_new_column_from_function(df_index, col_function: callable):
    # Use the index name to build a new pd.Series
    new_col = {}
    for name in tqdm(df_index):
        project_name, neuron_name = split_dataframe_name(name)

        col_value = col_function(project_name, neuron_name)
        # Add to dict with original index as key
        new_col[name] = col_value

    df_new_col = pd.Series(new_col)

    return df_new_col


def split_dataframe_name(dataframe_row_name):
    neuron_name = '_'.join(dataframe_row_name.split('_')[-2:])
    project_name = '_'.join(dataframe_row_name.split('_')[:-2])
    return project_name, neuron_name


def get_manual_annotation_from_project(all_projects, project_name, neuron_name,
                                       unknown_value='unknown',
                                       min_confidence=2):
    # Get the id'ed name for this project and this neuron, even if empty
    p = all_projects[project_name]
    mapping = p.dict_numbers_to_neuron_names()
    confidence = mapping.get(neuron_name, [0, 0])[1]
    if confidence >= min_confidence:
        col_value = mapping[neuron_name][0]
    else:
        col_value = unknown_value
    return col_value
