"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from wbfm.utils.visualization.behavior_comparison_plots import MarkovRegressionModel

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

    model = MarkovRegressionModel(_config['project_path'])
    model.plot_no_neuron_markov_model(to_save=True)
    model.plot_aic_feature_selected_neurons(to_save=True)
