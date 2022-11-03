"""
python path/to/this/script/4+extract_pixel_values_for_tracked_neurons.py with project_path=/path/to/your/project.yaml
"""

# Experiment tracking
import sacred
from sacred import Experiment

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.alternate_trace_calculations import save_alternate_trace_dataframes
from wbfm.utils.visualization.utils_segmentation import extract_list_of_pixel_values_from_config

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, also_save_trace_dataframes=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    print(project_path)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    extract_list_of_pixel_values_from_config(_config['project_path'])

    if _config['also_save_trace_dataframes']:
        project_data = ProjectData.load_final_project_data_from_config(_config['project_path'])
        save_alternate_trace_dataframes(project_data)
