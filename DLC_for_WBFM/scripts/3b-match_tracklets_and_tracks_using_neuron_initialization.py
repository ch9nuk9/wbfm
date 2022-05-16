"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
from sacred import Experiment

# main function

from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.neuron_matching.long_range_matching import global_track_matches_from_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, use_imputed_df=False, start_from_manual_matches=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_3b.log')

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run, _log):
    sacred.commands.print_config(_run)
    cfg = _config['cfg']

    DEBUG = _config['DEBUG']
    global_track_matches_from_config(cfg, DEBUG=DEBUG)
