"""
main
"""

from pathlib import Path

import matplotlib.pyplot as plt
# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

from wbfm.utils.projects.finished_project_data import load_all_projects_in_folder
# main function
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToUnivariateEncoding, MultiProjectBehaviorPlotter
from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets, get_summary_visualization_dir
from wbfm.utils.visualization.plot_traces import make_default_summary_plots_using_config

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
    folder_path = _config['project_path']
    if folder_path.lower() in ["gcamp", "gfp"]:
        # Save in common folder, but make a subfolder
        all_projects = load_paper_datasets(folder_path)
        output_folder = Path(get_summary_visualization_dir())
        output_folder = output_folder.joinpath(Path(folder_path).name)
        output_folder.mkdir(exist_ok=True)
    else:
        all_projects = load_all_projects_in_folder(folder_path)
        output_folder = Path(folder_path)

    behavior_plotter = MultiProjectBehaviorPlotter(all_projects, NeuronToUnivariateEncoding)
    print(f"Saving all output in folder: {output_folder}")

    # Single neuron plot options
    opt = dict(df_name='ratio', to_save=True, saving_folder=output_folder)

    # Signed speed
    behavior_plotter.paired_boxplot_overall_multi_dataset('ratio')
    fname = output_folder.joinpath("encoding_signed_speed.png")
    plt.savefig(fname)

    behavior_plotter.plot_model_prediction('ratio', to_save=True, use_multineuron=True)
    behavior_plotter.plot_model_prediction(use_multineuron=False, **opt)
    behavior_plotter.plot_sorted_correlations(**opt)

    # Absolute speed
    behavior_plotter.paired_boxplot_overall_multi_dataset('ratio', y_train='abs_speed')
    fname = output_folder.joinpath("encoding_absolute_speed.png")
    plt.savefig(fname)

    opt['y_train'] = 'abs_speed'
    behavior_plotter.plot_model_prediction(use_multineuron=True, **opt)
    behavior_plotter.plot_model_prediction(use_multineuron=False, **opt)
    behavior_plotter.plot_sorted_correlations(**opt)
    plt.close('all')

    # Curvature from leifer paper
    behavior_plotter.paired_boxplot_overall_multi_dataset('ratio', y_train='leifer_curvature')
    fname = output_folder.joinpath("encoding_leifer_curvature.png")
    plt.savefig(fname)

    opt['y_train'] = 'leifer_curvature'
    behavior_plotter.plot_model_prediction(use_multineuron=True, **opt)
    behavior_plotter.plot_model_prediction(use_multineuron=False, **opt)
    behavior_plotter.plot_sorted_correlations(**opt)
    plt.close('all')

    # Constructed pirouette state
    behavior_plotter.paired_boxplot_overall_multi_dataset('ratio', y_train='pirouette')
    fname = output_folder.joinpath("encoding_pirouette.png")
    plt.savefig(fname)

    opt['y_train'] = 'pirouette'
    behavior_plotter.plot_model_prediction(use_multineuron=True, **opt)
    behavior_plotter.plot_model_prediction(use_multineuron=False, **opt)
    behavior_plotter.plot_sorted_correlations(**opt)
    plt.close('all')

