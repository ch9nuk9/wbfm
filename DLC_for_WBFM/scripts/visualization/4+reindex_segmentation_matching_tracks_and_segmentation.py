

# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_using_config
# Experiment tracking
import sacred
from sacred import Experiment
from pathlib import Path

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None, out_fname=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)

    segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
    project_dir = Path(project_path).parent
    segment_fname = Path(project_dir).joinpath(segment_fname)
    segment_cfg = dict(load_config(segment_fname))

    traces_fname = Path(project_cfg['subfolder_configs']['traces'])
    traces_fname = Path(project_dir).joinpath(traces_fname)
    traces_cfg = dict(load_config(traces_fname))


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)

    this_config = _config.copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()
    this_config['project_dir'] = _config['project_dir']

    reindex_segmentation_using_config(this_config, _config['DEBUG'])
