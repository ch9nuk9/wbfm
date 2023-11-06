import sacred
from sacred import Experiment, SETTINGS
from wbfm.utils.projects.project_config_classes import rename_variable_in_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(num_volumes):
    pass


@ex.automain
def main(_config, _run, _log):
    sacred.commands.print_config(_run)

    num_volumes = _config['num_volumes']

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
