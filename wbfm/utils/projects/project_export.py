import os
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToUnivariateEncoding


def save_all_final_dataframes(project_data: Union[ProjectData, str]):
    """
    For now, saves several dataframes in a final_dataframes folder within the project

    Returns
    -------

    """
    # Will be local to the project
    output_folder = 'final_dataframes'
    project_data = ProjectData.load_final_project_data_from_config(project_data)
    project_config = project_data.project_config
    worm = project_data.worm_posture_class

    # Trace dataframes
    # Note: more efficient if these options align with the NeuronToUnivariateEncoding calculations below
    dict_all_trace_dfs = {}
    channel_modes = ['ratio', 'red', 'green']
    trace_opt = {'filter_mode': 'rolling_mean', 'min_nonnan': 0.9}
    for c in channel_modes:
        df = project_data.calc_default_traces(channel_mode=c, **trace_opt)
        dict_all_trace_dfs[c] = df

    residual_modes = ['pca', 'nmf']
    for r in residual_modes:
        key = f"ratio_residual_{r}"
        df = project_data.calc_default_traces(channel_mode='ratio', residual_mode=r, interpolate_nan=True, **trace_opt)
        dict_all_trace_dfs[key] = df

    df_all_traces = pd.concat(dict_all_trace_dfs.values(), axis=1, keys=dict_all_trace_dfs.keys())

    # Behavioral dataframe, fluorescence_fps
    model = NeuronToUnivariateEncoding(project_path=project_data)
    _ = model.all_dfs
    cols = ['summed_curvature', 'signed_speed_angular', 'signed_middle_body_speed']
    trace_len = project_data.num_frames
    df_behavior_single_columns = {c: model.unpack_behavioral_time_series_from_name(c, trace_len)[0] for c in cols}
    reversal_col = worm.beh_annotation(fluorescence_fps=True) == BehaviorCodes.REV
    df_behavior_single_columns['reversal'] = reversal_col.reset_index(drop=True)
    df_behavior = pd.DataFrame(df_behavior_single_columns)
    df_behavior.reversal.replace(np.nan, False, inplace=True)

    # Intermediate behavioral dataframe, because these are not multiindex, but curvature already is multiindex
    df_all_behavior_single_columns = pd.concat(df_behavior_single_columns.values(), axis=1, keys=df_behavior_single_columns.keys())

    df_behavior_multi_column = {}
    # Kymograph (curvature), fluorescence_fps and not
    df_curvature = worm.curvature(fluorescence_fps=True)
    df_curvature = df_curvature.reset_index(drop=True)
    df_curvature.columns = [f"segment_{i:03d}" for i in range(df_curvature.shape[1])]

    df_behavior_multi_column['curvature'] = df_curvature
    df_behavior_multi_column['behavior'] = df_all_behavior_single_columns

    df_all_behaviors = pd.concat(df_behavior_multi_column.values(), axis=1, keys=df_behavior_multi_column.keys())

    # Combine all into full multi-level dataframe and save
    df_final = pd.concat([df_all_traces, df_all_behaviors], axis=1, keys=['traces', 'behavior'])

    fname = os.path.join(output_folder, 'df_final.h5')
    project_config.save_data_in_local_project(df_final, fname)

    # fname = os.path.join(output_folder, 'behavior', 'df_curvature_high_fps.h5')
    # df_curvature = worm.curvature(fluorescence_fps=False)
    # df_curvature = df_curvature.reset_index(drop=True)
    # df_curvature.columns = [f"segment_{i:03d}" for i in range(df_curvature.shape[1])]
    #
    # project_config.h5_data_in_local_project(df_curvature, fname)

    #
    print("Finished saving all dataframes")

    return df_final


def read_dataframes_from_exported_folder(project_path: str) -> pd.DataFrame:
    fname = Path(project_path).parent.joinpath('final_dataframes/df_final.h5')
    df_final = pd.read_hdf(fname)

    return df_final


def concat_df_final_from_multiple_projects(all_project_paths: List[str]) -> pd.DataFrame:

    all_project_data = [ProjectData.load_final_project_data_from_config(p) for p in all_project_paths]
    all_dfs = [p.df_exported_format for p in all_project_data]
    keys = [p.more_shortened_name for p in all_project_data]

    df_multiproject = pd.concat(all_dfs, axis=1, keys=keys)
    return df_multiproject
