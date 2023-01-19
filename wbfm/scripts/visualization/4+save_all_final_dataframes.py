"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.project_export import save_all_final_dataframes

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None)


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    save_all_final_dataframes(_config['project_path'])
