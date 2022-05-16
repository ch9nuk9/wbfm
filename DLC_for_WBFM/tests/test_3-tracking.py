import importlib
from typing import Tuple

import pandas as pd
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.tests.global_vars_for_tests import project_path


def _load_tracks() -> Tuple[object, ModularProjectConfig]:
    cfg = ModularProjectConfig(project_path)
    track_cfg = cfg.get_tracking_config()

    fname = track_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
    df_tracks = pd.read_hdf(fname)
    return df_tracks, cfg


def test_pipeline_step_a():
    mod = importlib.import_module("DLC_for_WBFM.scripts.3a-track_using_superglue", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_b():
    mod = importlib.import_module("DLC_for_WBFM.scripts.3b-match_tracklets_and_tracks", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_saved_properly():
    df, _ = _load_tracks()
    assert type(df) == pd.DataFrame
