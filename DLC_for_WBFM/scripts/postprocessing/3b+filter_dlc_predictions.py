"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from DLC_for_WBFM.utils.traces.dlc_pipeline import filter_all_dlc_tracks
# main function
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

        is_abs = Path(track_fname).is_absolute()


@ex.automain
def make_dlc_labeled_videos(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config['track_cfg'].copy()
    filter_mode = _config['filter_mode']

    if _config['is_abs']:
        use_dlc_project_videos = False
    else:
        use_dlc_project_videos = True

    with safe_cd(_config['project_dir']):
        filter_all_dlc_tracks(this_config, use_dlc_project_videos=use_dlc_project_videos, filter_mode=filter_mode,
                              DEBUG=DEBUG)
