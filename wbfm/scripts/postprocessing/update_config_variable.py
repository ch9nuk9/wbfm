import sacred
from sacred import Experiment, SETTINGS
from wbfm.utils.projects.project_config_classes import update_variable_in_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, which_config="project", field_name=None, field_value=None, DEBUG=False)


@ex.config
def cfg():
    pass


@ex.automain
def main(_config, _run, _log):
    sacred.commands.print_config(_run)

    which_config = _config['which_config']
    field_name = _config['field_name']
    field_value = _config['field_value']

    # Hardcoded representing all changes
    variables_to_rename = {
        which_config: {
            field_name: field_value
        }
    }
    project_path = _config['project_path']

    update_variable_in_config(project_path, variables_to_rename)
