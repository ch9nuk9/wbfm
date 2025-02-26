import os
import os.path as osp
import pathlib
import shutil
import typing
from contextlib import contextmanager
from datetime import datetime
from os import path as osp
from pathlib import Path

from wbfm.utils.external.utils_yaml import edit_config, load_config
from wbfm.utils.general.utils_filenames import get_location_of_new_project_defaults


#####################
# Filename utils
#####################


def get_project_name(basename=None, experimenter='', task='') -> str:
    # Use current time
    if basename is None:
        project_name = datetime.now().strftime("%Y_%m_%d")
    else:
        project_name = basename
    if task is not None and task != '':
        project_name = f"{task}-" + project_name
    if experimenter is not None and experimenter != '':
        project_name = f"{experimenter}-" + project_name

    return project_name


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


def update_project_config_path(abs_dir_name, project_config_updates=None):
    dest_fname = 'project_config.yaml'
    project_fname = osp.join(abs_dir_name, dest_fname)
    project_fname = Path(project_fname).resolve()
    if project_config_updates is None:
        project_config_updates = dict(project_path=str(project_fname))
    edit_config(str(project_fname), project_config_updates)
    return project_fname


def update_snakemake_config_path(abs_dir_name):
    snakemake_fname = osp.join(abs_dir_name, 'snakemake', 'snakemake_config.yaml')
    snakemake_updates = {'project_dir': str(abs_dir_name)}
    edit_config(snakemake_fname, snakemake_updates)


def update_nwb_config_path(abs_dir_name, nwb_filename):
    nwb_fname = osp.join(abs_dir_name, 'nwb', 'nwb_config.yaml')
    nwb_updates = {'nwb_filename': str(nwb_filename)}
    edit_config(nwb_fname, nwb_updates)
