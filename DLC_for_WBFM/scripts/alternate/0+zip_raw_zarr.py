"""
"""

# main function
import os
import subprocess
from pathlib import Path

# Experiment tracking
import sacred

from DLC_for_WBFM.utils.external.utils_zarr import zip_raw_data_zarr
from DLC_for_WBFM.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

from DLC_for_WBFM.pipeline.project_initialization import write_data_subset_using_config, zip_zarr_using_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    zip_zarr_using_config(_config['cfg'])

    print("Finished.")

