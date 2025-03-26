import os
import shutil

from wbfm.utils.projects.finished_project_data import ProjectData


def add_neuropal_to_project(project_path, neuropal_path, copy_data=True):
    """
    Adds a neuropal dataset to an existing project, without analyzing it.

    Parameters
    ----------
    project
    neuropal_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data(project_path)
    neuropal_config = project_data.project_config.initialize_neuropal_subproject()
    target_dir = neuropal_config.absolute_subfolder

    # Move or copy the neuropal data to the project directory
    shutil.copy(neuropal_path, target_dir)
    if not copy_data:
        if os.path.isfile(neuropal_path):
            os.remove(neuropal_path)
        else:
            shutil.rmtree(neuropal_path)

    # Update the config file with the new data path
    neuropal_data_path = os.path.join(target_dir, os.path.basename(neuropal_path))
    neuropal_config.config['neuropal_data_path'] = neuropal_data_path
    neuropal_config.update_self_on_disk()
