"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None,)


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    model = NeuronToMultivariateEncoding(_config['project_path'])

    model.plot_correlation_of_examples(to_save=True)
    model.plot_correlation_histograms(to_save=True)
    model.plot_phase_difference(to_save=True)
    model.plot_histogram_difference_after_ratio(to_save=True)
    model.plot_paired_boxplot_difference_after_ratio(to_save=True)
