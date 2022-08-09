"""
The top level function for producing training data via feature-based tracking
"""
# Experiment tracking
import sacred
from sacred import Experiment
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.pipeline.tracklets import build_frame_pairs_using_superglue_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    check_all_needed_data_for_step(cfg, 2)

    if not DEBUG:
        using_monkeypatch()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_config = _config['cfg']

    build_frame_pairs_using_superglue_using_config(project_config, DEBUG=False)
