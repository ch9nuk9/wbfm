import importlib
import logging
from wbfm.tests.unit_tests.global_vars_for_tests import project_path


def test_pipeline_step():
    logging.info("Note: must be in the utils_fdnc (pytorch enabled) conda environment")
    # Run the sacred experiment from the actual script
    mod = importlib.import_module("wbfm.scripts.pipeline_alternate.3-track_using_fdnc", package="wbfm")

    config_updates = {'project_path': project_path, 'DEBUG': False}

    mod.ex.run(config_updates=config_updates)
