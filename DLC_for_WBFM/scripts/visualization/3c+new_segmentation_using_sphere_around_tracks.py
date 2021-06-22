

# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config
from DLC_for_WBFM.utils.visualization.utils_segmentation import create_spherical_segmentation
# Experiment tracking
import sacred
from sacred import Experiment
from pathlib import Path

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None, sphere_radius=5.0, out_fname=None)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)

    track_fname = Path(project_cfg['subfolder_configs']['tracking'])
    project_dir = Path(project_path).parent
    track_fname = Path(project_dir).joinpath(track_fname)
    track_cfg = dict(load_config(track_fname))

    segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
    project_dir = Path(project_path).parent
    segment_fname = Path(project_dir).joinpath(segment_fname)
    segment_cfg = dict(load_config(segment_fname))


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)

    this_config = _config.copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    sphere_radius = _config['sphere_radius']

    create_spherical_segmentation(this_config, sphere_radius)