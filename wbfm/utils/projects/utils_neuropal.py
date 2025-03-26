import os
import shutil
from pathlib import Path

from imutils import MicroscopeDataReader

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

    # Make sure we have the path to the folder, not just the file
    if os.path.isfile(neuropal_path):
        neuropal_dir = os.path.dirname(neuropal_path)
    else:
        neuropal_dir = neuropal_path

    # Check: make sure this is readable by Lukas' reader
    # Note that we need the ome.tif file within the neuropal directory
    for file in Path(neuropal_dir).iterdir():
        if file.is_dir():
            continue
        elif str(file).endswith('ome.tif'):
            neuropal_path = str(file)
            break
    else:
        raise FileNotFoundError(f'Could not find the ome.tif file in {neuropal_dir}')
    _ = MicroscopeDataReader(neuropal_path)

    # Move or copy all contents from the neuropal folder project directory
    if copy_data:
        shutil.copytree(neuropal_dir, target_dir, dirs_exist_ok=True)
    else:
        shutil.move(neuropal_dir, target_dir)

    # Update the config file with the new data path
    neuropal_data_path = os.path.join(target_dir, os.path.basename(neuropal_path))
    neuropal_config.config['neuropal_data_path'] = neuropal_config.unresolve_absolute_path(neuropal_data_path)
    neuropal_config.update_self_on_disk()
