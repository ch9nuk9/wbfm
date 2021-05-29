"""
"""

# main function
from DLC_for_WBFM.utils.projects.utils_data_subsets import segment_local_data_subset
# Experiment tracking
import sacred
from sacred import Experiment

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, out_fname=None)

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    segment_local_data_subset(_config['project_path'], _config['out_fname'])