import importlib
from DLC_for_WBFM.tests.global_vars_for_tests import project_path
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


def test_pipeline_step_a():
    mod = importlib.import_module("DLC_for_WBFM.scripts.4a-match_tracks_and_segmentation", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_b():
    mod = importlib.import_module("DLC_for_WBFM.scripts.4b-reindex_segmentation_full", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_c():
    mod = importlib.import_module("DLC_for_WBFM.scripts.4c-extract_full_traces", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_all_loaded():
    project_dat = ProjectData.load_final_project_data_from_config(project_path)

    assert project_dat.red_traces is not None
    assert project_dat.green_traces is not None
