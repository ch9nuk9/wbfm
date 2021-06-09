

# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from segmentation.util.utils_pipeline import recalculate_metadata_from_config
# Experiment tracking
import sacred
from sacred import Experiment
from pathlib import Path
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, out_fname=None)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
    project_dir = Path(project_path).parent
    segment_fname = Path(project_dir).joinpath(segment_fname)
    segment_cfg = dict(load_config(segment_fname))


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)

    this_config = _config['segment_cfg'].copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    with safe_cd(_config['project_dir']):
        recalculate_metadata_from_config(this_config)