"""
Usage:

python path/to/this/script/4+extract_pixel_values_for_tracked_neurons.py with project_path=/path/to/your/project.yaml
"""

# Experiment tracking
import sacred
from sacred import Experiment
from wbfm.utils.visualization.utils_segmentation import extract_list_of_pixel_values_from_config

# Initialize sacred experiment

ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    extract_list_of_pixel_values_from_config(_config['project_path'])
