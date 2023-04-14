import importlib
from typing import Tuple

import numpy as np
import pandas as pd

from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.tests.unit_tests.global_vars_for_tests import project_path


def _load_training_data() -> Tuple[object, ModularProjectConfig]:
    cfg = ModularProjectConfig(project_path)
    training_cfg = cfg.get_training_config()

    fname = training_cfg.resolve_relative_path_from_config('df_3d_tracklets')
    df_tracks = pd.read_hdf(fname)
    return df_tracks, cfg


def test_pipeline_step_a():
    mod = importlib.import_module("wbfm.scripts.2a-build_frame_objects", package="wbfm")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_b():
    mod = importlib.import_module("wbfm.scripts.2b-match_adjacent_volumes", package="wbfm")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_c():
    mod = importlib.import_module("wbfm.scripts.2c-postprocess_matches_to_tracklets", package="wbfm")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_saved_properly():
    df, _ = _load_training_data()
    assert type(df) == pd.DataFrame


def test_finds_matches():
    cfg = ModularProjectConfig(project_path)
    df, _ = _load_training_data()

    expected_len = cfg.config['dataset_params']['num_frames'] - 1
    assert len(df) == expected_len

    minimum_expected_matches = 110
    assert df.shape[1] > minimum_expected_matches


# @pytest.mark.xfail(reason="Known incorrect segmentation leads to unreasonable z changes")
def test_reasonable_z():
    df, cfg = _load_training_data()

    max_z_delta = cfg.get_training_config().config['pairwise_matching_params']['z_threshold']

    df_delta = df.diff()[1:]
    all_neurons = get_names_from_df(df)

    for n in all_neurons:
        z_delta_series = df_delta[n]['z']
        is_small = np.abs(z_delta_series <= max_z_delta)
        is_nan = np.isnan(z_delta_series)
        assert all(np.logical_or(is_small, is_nan))


# if __name__ == "__main__":
#     test_pipeline_step()
