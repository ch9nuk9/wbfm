"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.utils_export_videos import save_video_of_heatmap_with_behavior

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

    proj_dat = ProjectData.load_final_project_data_from_config(_config['project_path'])
    proj_dat.verbose = 0

    save_video_of_heatmap_with_behavior(proj_dat)
