import sacred
from sacred import Experiment
from wbfm.utils.projects.project_config_classes import rename_variable_in_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg():
    pass


@ex.automain
def main(_config, _run, _log):
    sacred.commands.print_config(_run)

    # Hardcoded representing all changes
    variables_to_rename = dict(
        project=dict(
            dataset_params=dict(
                num_frames=num_volumes
            )
        )
    )
    project_path = _config['project_path']

    rename_variable_in_config(project_path, variables_to_rename)
