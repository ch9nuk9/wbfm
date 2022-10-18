import os
import os.path as osp
import pathlib
import shutil
import typing
from contextlib import contextmanager
from datetime import datetime
from os import path as osp
from pathlib import Path

from ruamel.yaml import YAML
from wbfm.utils.projects.utils_filenames import get_location_of_new_project_defaults


#####################
# Filename utils
#####################


def get_project_name(_config: dict) -> str:
    # Use current time
    project_name = datetime.now().strftime("%Y_%m_%d")
    exp = _config.get('experimenter', '')
    task = _config.get('task_name', '')
    if task is not None and task != '':
        project_name = f"{task}-" + project_name
    if exp is not None and exp != '':
        project_name = f"{exp}-" + project_name

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

    assert project_path.endswith('.yaml'), "Must pass a valid config file"

    project_dir = Path(project_path).parent
    if verbose >= 1:
        print(f"Cleaning project {project_dir}")

    # Get a list of all files that should be present, relative to the project directory
    src = get_location_of_new_project_defaults()
    initial_fnames = list(Path(src).rglob('**/*'))
    if len(initial_fnames) == 0:
        print("Found no initial files, probably running this from the wrong directory")
        raise FileNotFoundError

    # Convert them to relative
    initial_fnames = {str(fname.relative_to(src)) for fname in initial_fnames}
    if verbose >= 3:
        print(f"Found initial files: {initial_fnames}")

    # Also get the filenames of the target folder
    target_fnames = list(Path(project_dir).rglob('**/*'))
    if verbose >= 3:
        print(f"Found target files: {target_fnames}")

    # Check each target fname, and if it is not in the initial set, delete it
    if dryrun:
        print("DRYRUN (nothing actually deleted)")
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

    # Also remove the created directories, which are .zarr
    for fname in target_fnames:
        if fname.is_dir() and str(fname).endswith('.zarr'):
            shutil.rmtree(fname)

    if dryrun:
        print("DRYRUN (nothing actually deleted)")
        print("If you want to really delete things, then use 'dryrun=False' in the command line")


def make_project_like(project_path: str, target_directory: str, new_project_name: str = None, verbose=1):
    """Copy all config files from a project, i.e. only the files that would exist in a new project"""

    assert project_path.endswith('.yaml'), "Must pass a valid config file"
    assert os.path.exists(target_directory), "Must pass a folder that exists"

    project_dir = Path(project_path).parent
    if new_project_name is None:
        new_project_name = project_dir.name
    target_project_name = Path(target_directory).joinpath(new_project_name)
    if verbose >= 1:
        print(f"Copying project {project_dir}")

    # Get a list of all files that should be present, relative to the project directory
    src = get_location_of_new_project_defaults()
    initial_fnames = list(Path(src).rglob('**/*'))
    if len(initial_fnames) == 0:
        print("Found no initial files, probably running this from the wrong directory")
        raise FileNotFoundError

    # Convert them to relative
    initial_fnames = {str(fname.relative_to(src)) for fname in initial_fnames}
    if verbose >= 3:
        print(f"Found initial files: {initial_fnames}")

    # Also get the filenames of the target folder
    target_fnames = list(Path(project_dir).rglob('**/*'))
    if verbose >= 3:
        print(f"Found target files: {target_fnames}")

    # Check each initial project fname, and if it is in the initial set, copy it
    for fname in target_fnames:
        if fname.is_dir():
            continue
        rel_fname = fname.relative_to(project_dir)
        new_fname = target_project_name.joinpath(rel_fname)
        if str(rel_fname) in initial_fnames:
            os.makedirs(new_fname.parent, exist_ok=True)
            shutil.copy(fname, new_fname)

            if verbose >= 1:
                print(f"Copying {rel_fname}")
        elif verbose >= 2:
            print(f"Not copying {rel_fname}")

    # Update the copied project config with the new dest folder
    update_project_config_path(_config, abs_dir_name)

    # Also update the snakemake file with the project directory
    update_snakemake_config_path(abs_dir_name)


def update_project_config_path(_config, abs_dir_name):
    dest_fname = 'project_config.yaml'
    project_fname = osp.join(abs_dir_name, dest_fname)
    project_fname = Path(project_fname).resolve()
    edit_config(str(project_fname), _config)


def update_snakemake_config_path(abs_dir_name):
    snakemake_fname = osp.join(abs_dir_name, 'snakemake', 'config.yaml')
    snakemake_updates = {'project_dir': abs_dir_name}
    edit_config(snakemake_fname, snakemake_updates)
