from functools import partial
from typing import Dict

import pandas as pd
from sklearn.model_selection import KFold

from wbfm.gui.utils.utils_dash import save_folder_for_two_dataframe_dashboard
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.behavior_comparison_plots import MultiProjectBehaviorPlotter, NeuronToUnivariateEncoding


def build_all_gui_dfs_speed_encoding(all_projects_gcamp: Dict[str, ProjectData],
                                     all_projects_gfp: Dict[str, ProjectData],
                                     output_folder: str = None,
                                     trace_options=None,
                                     **kwargs):

    # Use same trace options for all plots
    if trace_options is None:
        trace_options = {}
    opt = dict(interpolate_nan=False,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=False,  # This takes a long time!
               channel_mode='dr_over_r_50')
    opt.update(trace_options)

    encoder_opt = dict(df_name=opt['channel_mode'], y_train='signed_middle_body_speed')
    constructor_kwargs = dict(df_kwargs=opt, dataframes_to_load=[opt['channel_mode']])

    cv_factory = partial(KFold, n_splits=3)

    # Get summary dataframes
    df_opt = dict(encoder_opt=encoder_opt, constructor_kwargs=constructor_kwargs, cv_factory=cv_factory)
    df_summary_gcamp, df_prediction_gcamp, df_raw_gcamp = calculate_all_dfs_using_encoder(all_projects_gcamp,
                                                                                          genotype='gcamp',
                                                                                          **df_opt)
    df_summary_gfp, df_prediction_gfp, df_raw_gfp = calculate_all_dfs_using_encoder(all_projects_gfp, genotype='gfp',
                                                                                    **df_opt)

    # Concatenate all dictionaries into single dataframes
    df_summary = pd.concat([df_summary_gcamp, df_summary_gfp])
    df_pred = pd.concat([df_prediction_gcamp, df_prediction_gfp], axis=1)
    df_raw = pd.concat([df_raw_gcamp, df_raw_gfp], axis=1)
    raw_dfs = {'speed': df_raw, 'prediction': df_pred}

    # Save
    save_folder_for_two_dataframe_dashboard(output_folder, df_summary, raw_dfs)


def calculate_all_dfs_using_encoder(all_projects, genotype='gcamp', only_model_single_state=None,
                                    encoder_opt=None, constructor_kwargs=None, cv_factory=None):
    multi_encoder_gcamp = MultiProjectBehaviorPlotter(all_projects=all_projects, constructor_kwargs=constructor_kwargs,
                                                      class_constructor=NeuronToUnivariateEncoding)
    multi_encoder_gcamp.set_for_all_classes({'cv_factory': cv_factory})

    _encoder_opt = encoder_opt.copy()
    _encoder_opt['only_model_single_state'] = only_model_single_state
    df_summary_gcamp = multi_encoder_gcamp.calc_dataset_summary_df(**_encoder_opt)
    df_prediction_gcamp = multi_encoder_gcamp.calc_prediction_or_raw_df(**_encoder_opt, prediction_not_raw=True)
    df_raw_gcamp = multi_encoder_gcamp.calc_prediction_or_raw_df(**_encoder_opt, prediction_not_raw=False)

    # Concatenate all dictionaries into single dataframes
    df_summary_gcamp = pd.concat(df_summary_gcamp)
    df_summary_gcamp['genotype'] = genotype
    df_summary_gcamp.index = df_summary_gcamp.index.droplevel(1)

    df_prediction_gcamp = pd.concat(df_prediction_gcamp, axis=1)
    df_prediction_gcamp.columns = df_prediction_gcamp.columns.droplevel(1)

    df_raw_gcamp = pd.concat(df_raw_gcamp, axis=1)
    df_raw_gcamp.columns = df_raw_gcamp.columns.droplevel(1)

    return df_summary_gcamp, df_prediction_gcamp, df_raw_gcamp
