"""
Postprocessing script to delete all analysis files in a project directory.

This script is useful when you want to re-run the analysis from scratch, or something crashed and can't be recovered

Usage:
```bash
python wbfm/scripts/postprocessing/delete_analysis_files.py with project_path={PATH}/project_config.yaml dryrun=True
```

Note: dryrun=True will not actually delete anything, but will print out what would be deleted

"""

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.utils_project import delete_all_analysis_files

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, dryrun=True, DEBUG=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    dryrun = _config['dryrun']
    project_path = _config['project_path']

    delete_all_analysis_files(project_path, dryrun=dryrun, verbose=2)
