import importlib
from wbfm.tests.unit_tests.global_vars_for_tests import project_path


def test_segmentation():
    # Run the sacred experiment from the actual script
    mod = importlib.import_module("wbfm.scripts.1-segment_video", package="wbfm")

    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)
