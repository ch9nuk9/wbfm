"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS

# main function
from DLC_for_WBFM.gui.utils.manual_annotation import create_manual_correction_gui
from DLC_for_WBFM.utils.pipeline.traces_pipeline import get_traces_from_3d_tracks_using_config
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        trace_fname = Path(project_cfg['subfolder_configs']['traces'])
        trace_cfg = dict(load_config(trace_fname))
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))
        seg_fname = Path(project_cfg['subfolder_configs']['segmentation'])
        segment_cfg = dict(load_config(seg_fname))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config.copy()

    with safe_cd(_config['project_dir']):
        create_manual_correction_gui(this_config, DEBUG=DEBUG)
