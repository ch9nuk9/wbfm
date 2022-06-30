import os
import os.path as osp
import pathlib
import typing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML


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
    """Generic overwriting, based on DLC. Will create new file if one isn't found"""

    if DEBUG:
        print(f"Editing config file at: {config_fname}")
    if Path(config_fname).exists():
        cfg = load_config(config_fname)
    else:
        cfg = {}
        print(f"Config file not found, creating new one")

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


def delete_all_analysis_files(project_path: str, dryrun=False, verbose=2):
    """Deletes all files produced by analysis, reverting a project to only the files present in a raw default project"""

    project_dir = Path(project_path).parent
    if verbose >= 1:
        print(f"Cleaning project {project_dir}")

    # Get a list of all files that should be present, relative to the project directory
    # Note: assumes that you are executing from the main DLC_for_WBFM folder
    src = 'new_project_defaults'
    initial_fnames = list(Path(src).rglob('**/*'))
    # initial_fnames = get_abs_filenames_recursive(src)

    # Convert them to relative
    initial_fnames = {str(fname.relative_to(src)) for fname in initial_fnames}
    if verbose >= 2:
        print(f"Found initial files: {initial_fnames}")

    # Also get the filenames of the target folder
    # target_fnames = get_abs_filenames_recursive(project_dir)
    target_fnames = list(Path(project_dir).rglob('**/*'))
    if verbose >= 2:
        print(f"Found target files: {target_fnames}")

    # Check each target fname, and if it is not in the initial set, delete it
    if dryrun:
        print("DRYRUN")
    for fname in target_fnames:
        if fname.is_dir():
            continue
        if str(fname.relative_to(project_dir)) in initial_fnames:
            if verbose >= 1:
                print(f"Keeping {fname.relative_to(project_dir)}")
        elif verbose >= 2:
            print(f"Deleting {fname.relative_to(project_dir)}")
            if not dryrun:
                os.remove(fname)

    # Also make sure any missing files are now present
    # for fname in initial_fnames:
    #     target_fname = Path(fname).relative_to()
