"""
"""

# main function
import os

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames_and_save
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project_status import check_zarr_file_integrity

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None,
              DEBUG=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    check_zarr_file_integrity(_config['project_path'])
