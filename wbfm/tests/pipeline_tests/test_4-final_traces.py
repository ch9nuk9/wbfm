import importlib
from wbfm.tests.unit_tests.global_vars_for_tests import project_path
from wbfm.utils.projects.finished_project_data import ProjectData


def test_pipeline_step_a():
    mod = importlib.import_module("wbfm.scripts.4-make_final_traces", package="wbfm")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_all_loaded():
    project_dat = ProjectData.load_final_project_data_from_config(project_path)

    assert project_dat.red_traces is not None
    assert project_dat.green_traces is not None
