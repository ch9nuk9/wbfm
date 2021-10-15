import importlib
from DLC_for_WBFM.tests.global_vars_for_tests import project_path


def test_segmentation():
    # Run the sacred experiment from the actual script
    mod = importlib.import_module("DLC_for_WBFM.scripts.1-segment_video", package="DLC_for_WBFM")

    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)
