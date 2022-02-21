import os
import os.path as osp
import pathlib
import typing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import copytree

from DLC_for_WBFM.utils.projects.utils_filenames import get_sequential_filename
from ruamel.yaml import YAML


def build_project_structure(_config: dict) -> None:
    # parent_folder = Path(_config['project_dir']).resolve()
    parent_folder = _config['project_dir']
    rel_dir_name = get_project_name(_config)

    # Build copied folder structure
    abs_dir_name = osp.join(parent_folder, rel_dir_name)
    abs_dir_name = get_sequential_filename(abs_dir_name)
    print(f"Building new project at: {abs_dir_name}")

    src = 'new_project_defaults'
    copytree(src, abs_dir_name)

    # Update the copied project config with the new dest folder
    dest_fname = 'project_config.yaml'
    project_fname = osp.join(abs_dir_name, dest_fname)
    project_fname = Path(project_fname).resolve()
    edit_config(str(project_fname), _config)


#####################
# Filename utils
#####################

def get_project_name(_config: dict) -> str:
    # Use current time
    project_name = datetime.now().strftime("%Y_%m_%d")
    exp = _config['experimenter']
    task = _config['task_name']
    project_name = f"{exp}-{task}-" + project_name
    return project_name


#####################
# config utils
#####################


def edit_config(config_fname: typing.Union[str, pathlib.Path], edits: dict, DEBUG: bool = False) -> dict:
    """Generic overwriting, based on DLC"""

    if DEBUG:
        print(f"Editing config file at: {config_fname}")
    cfg = load_config(config_fname)
    if DEBUG:
        print(f"Initial config: {cfg}")
        print(f"Edits: {edits}")

    for k, v in edits.items():
        cfg[k] = v

    with open(config_fname, "w") as f:
        YAML().dump(cfg, f)

    return cfg


def load_config(config_fname: typing.Union[str, pathlib.Path]) -> dict:
    assert osp.exists(config_fname), f"{config_fname} not found!"

    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    return cfg

#####################
# Synchronizing config files
#####################


def get_subfolder(project_path, subfolder):
    project_cfg = load_config(project_path)
    return Path(project_cfg['subfolder_configs'][subfolder]).parent


def get_project_of_substep(subfolder_path):
    return Path(Path(subfolder_path).parent).parent


@contextmanager
def safe_cd(newdir: typing.Union[str, pathlib.Path]) -> None:
    """
    Safe change directory that switches back

    @param newdir:
    """
    # https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory/24176022#24176022
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)