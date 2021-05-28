"""
"""

# main function
from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
# Experiment tracking
import sacred
from sacred import Experiment

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, out_fname=None, tiff_not_zarr=True, pad_to_align_with_original=False)

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    write_data_subset_from_config(_config['project_path'], _config['out_fname'])