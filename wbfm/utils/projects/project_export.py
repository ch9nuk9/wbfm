import os
from pathlib import Path
from typing import Union

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
    trace_opt = {'filter_mode': 'rolling_mean'}
    for c in channel_modes:
        fname = os.path.join(output_folder, 'traces', f'df_traces_{c}.h5')
        if os.path.exists(project_config.resolve_relative_path(fname)):
            continue
        df = project_data.calc_default_traces(channel_mode=c, **trace_opt)
        project_config.h5_data_in_local_project(df, fname)

    # Behavioral dataframe
    fname = os.path.join(output_folder, 'behavior', 'df_behavior.h5')
    if not os.path.exists(project_config.resolve_relative_path(fname)):
        model = NeuronToUnivariateEncoding(project_path=project_data)
        _ = model.all_dfs
        cols = ['signed_speed', 'abs_speed', 'leifer_curvature', 'signed_speed_smoothed', 'signed_speed_angular']
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


def read_dataframes_from_exported_folder(data_folder):
    _df_curvature, _df_behavior, _df_traces = None, None, None
    for subfolder in Path(data_folder).iterdir():
        if subfolder.name == 'traces':
            for file in Path(subfolder).iterdir():
                if 'df_traces_ratio' in file.name:
                    _df_traces = pd.read_hdf(file)
        elif subfolder.name == 'behavior':
            for file in Path(subfolder).iterdir():
                if 'df_behavior' in file.name:
                    _df_behavior = pd.read_hdf(file)
                elif 'df_curvature' in file.name:
                    _df_curvature = pd.read_hdf(file)
    assert (_df_curvature is not None and _df_behavior is not None and _df_traces is not None), \
        "Did not find all files!"
    return _df_behavior, _df_curvature, _df_traces
