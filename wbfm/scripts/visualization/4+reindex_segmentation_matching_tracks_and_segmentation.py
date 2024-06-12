# main function
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from wbfm.utils.external.utils_yaml import load_config
from wbfm.pipeline.traces import reindex_segmentation_using_config

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
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
