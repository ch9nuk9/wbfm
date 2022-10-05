
# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.utils_project import make_project_like

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, target_directory=None, DEBUG=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    target_directory = _config['target_directory']
    project_path = _config['project_path']

    make_project_like(project_path, target_directory=target_directory, verbose=1)
