import os
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import pandas as pd

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
    channel_modes = ['ratio', 'red', 'green']
    residual_modes = ['pca', 'nmf']
    trace_opt = {'filter_mode': 'rolling_mean', 'min_nonnan': 0.9}
    for c in channel_modes:
        fname = os.path.join(output_folder, 'traces', f'df_traces-{c}.h5')
        if os.path.exists(project_config.resolve_relative_path(fname)):
            continue
        df = project_data.calc_default_traces(channel_mode=c, **trace_opt)
        project_config.h5_data_in_local_project(df, fname)

    for r in residual_modes:
        fname = os.path.join(output_folder, 'traces', f'df_traces-ratio_residual_subtracted_{r}.h5')
        if os.path.exists(project_config.resolve_relative_path(fname)):
            continue
        df = project_data.calc_default_traces(channel_mode='ratio', residual_mode=r, interpolate_nan=True, **trace_opt)
        project_config.h5_data_in_local_project(df, fname)

    # Behavioral dataframe
    fname = os.path.join(output_folder, 'behavior', 'df_behavior.h5')
    if not os.path.exists(project_config.resolve_relative_path(fname)):
        model = NeuronToUnivariateEncoding(project_path=project_data)
        _ = model.all_dfs
        cols = ['signed_stage_speed', 'abs_stage_speed', 'leifer_curvature',
                'signed_stage_speed_smoothed', 'signed_speed_angular', 'signed_middle_body_speed']
        trace_len = project_data.num_frames
        df_behavior_dict = {c: model.unpack_behavioral_time_series_from_name(c, trace_len)[0] for c in cols}
        df_behavior_dict['reversal'] = (worm.beh_annotation(fluorescence_fps=True) == 1).reset_index(drop=True)
        df_behavior = pd.DataFrame(df_behavior_dict)
        df_behavior.reversal.replace(np.nan, False, inplace=True)

        project_config.h5_data_in_local_project(df_behavior, fname)

    # Kymograph (curvature)
    fname = os.path.join(output_folder, 'behavior', 'df_curvature.h5')
    if not os.path.exists(project_config.resolve_relative_path(fname)):
        df_curvature = worm.curvature(fluorescence_fps=True)
        df_curvature = df_curvature.reset_index(drop=True)
        df_curvature.columns = [f"segment_{i:03d}" for i in range(df_curvature.shape[1])]

        project_config.h5_data_in_local_project(df_curvature, fname)

    #
    print("Finished saving all dataframes")


def read_dataframes_from_exported_folder(project_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_folder = Path(project_path).parent.joinpath('final_dataframes')
    dict_of_dataframes = dict(traces={}, behavior={})
    for subfolder in Path(data_folder).iterdir():
        if subfolder.name == 'traces':
            for file in Path(subfolder).iterdir():
                if 'df_traces' in file.name:
                    # Assumes filename like df_traces-ratio.h5 or df_traces-ratio_residual.h5
                    key = file.name.split('-')[1].split('.')[0]
                    dict_of_dataframes['traces'][key] = pd.read_hdf(file)
        elif subfolder.name == 'behavior':
            for file in Path(subfolder).iterdir():
                if 'df_behavior' in file.name:
                    dict_of_dataframes['behavior']['behavior'] = pd.read_hdf(file)
                elif 'df_curvature' in file.name:
                    dict_of_dataframes['behavior']['curvature'] = pd.read_hdf(file)

    return dict_of_dataframes
