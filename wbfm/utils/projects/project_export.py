import os
from types import Union

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

    # Trace dataframes
    # Note: more efficient if these options align with the NeuronToUnivariateEncoding calculations below
    channel_modes = ['ratio', 'red', 'green']
    trace_opt = {'filter_mode': 'rolling_mean'}
    for c in channel_modes:
        df = project_data.calc_default_traces(channel_mode=c, **trace_opt)
        fname = os.path.join(output_folder, f'df_traces_{c}.h5')
        project_data.project_config.h5_data_in_local_project(df, fname)

    # Behavioral dataframe
    model = NeuronToUnivariateEncoding(project_path=project_data)
    _ = model.all_dfs
    cols = ['signed_speed', 'abs_speed', 'leifer_curvature', 'signed_speed_smoothed', 'signed_speed_angular']
    trace_len = len(df)
    df_behavior_dict = {c: model.unpack_behavioral_time_series_from_name(c, trace_len)[0] for c in cols}
    worm = project_data.worm_posture_class
    df_behavior_dict['reversal'] = (worm.beh_annotation(fluorescence_fps=True) == 1).reset_index(drop=True)
    df_behavior = pd.DataFrame(df_behavior_dict)
    df_behavior.reversal.replace(np.nan, False, inplace=True)

    fname = os.path.join(output_folder, 'df_behavior.h5')
    project_data.project_config.h5_data_in_local_project(df_behavior, fname)

    # Kymograph (curvature)
    df_curvature = worm.curvature(fluorescence_fps=True)
    df_curvature = df_curvature.reset_index(drop=True)
    df_curvature.columns = [f"segment_{i:03d}" for i in range(df_curvature.shape[1])]

    fname = os.path.join(output_folder, 'df_curvature.h5')
    project_data.project_config.h5_data_in_local_project(df_curvature, fname)
