"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, \
    make_summary_interactive_heatmap_with_kymograph

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

    project_cfg = ModularProjectConfig(_config['project_path'])
    make_summary_interactive_heatmap_with_pca(project_cfg, to_show=False, to_save=True)
    make_summary_interactive_heatmap_with_kymograph(project_cfg, to_show=False, to_save=True)
