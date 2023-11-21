"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

from wbfm.utils.projects.finished_project_data import ProjectData
# main function
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, \
    make_summary_interactive_kymograph_with_behavior

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

    trace_opt = dict(use_paper_options=True)

    project_cfg = ModularProjectConfig(_config['project_path'])
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    project_data.use_physical_time = True

    make_summary_interactive_heatmap_with_pca(project_data, to_show=False, to_save=True, trace_opt=trace_opt)
    make_summary_interactive_kymograph_with_behavior(project_data, to_show=False, to_save=True, trace_opt=trace_opt,
                                                     crop_x_axis=False)
