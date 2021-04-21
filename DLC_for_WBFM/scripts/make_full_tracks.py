"""
The top level function for producing training data via feature-based tracking
"""

from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from DLC_for_WBFM.utils.pipeline.DLC_utils import make_3d_tracks_from_stack
# Experiment tracking
import sacred
from sacred import Experiment

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
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config['track_cfg'].copy()

    with safe_cd(_config['project_dir']):
        make_3d_tracks_from_stack(this_config, DEBUG=DEBUG)
