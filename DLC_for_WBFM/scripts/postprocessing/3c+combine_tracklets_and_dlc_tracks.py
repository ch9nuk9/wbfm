"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from DLC_for_WBFM.utils.postprocessing.combine_tracklets_and_DLC_tracks import \
    combine_all_dlc_and_tracklet_coverings_from_config
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, filter_mode='arima', DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))


@ex.automain
def combine_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config['track_cfg'].copy()
    this_config['project_dir'] = _config['project_dir']

    with safe_cd(_config['project_dir']):
        combine_all_dlc_and_tracklet_coverings_from_config(this_config, DEBUG=DEBUG)
