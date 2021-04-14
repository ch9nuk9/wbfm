import os.path as osp
from datetime import datetime
from pathlib import Path
from shutil import copyfile, copytree
from ruamel.yaml import YAML


def build_project_structure(_config):

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

    # Update certain subconfigs to have absolute paths
    subfolder = Path(abs_dir_name).joinpath('segmentation')
    to_edit = subfolder.joinpath('segment_config.yaml')
    new_abs_path = subfolder.joinpath('preprocessing_config.yaml')
    updates = {'preprocessing_config': str(new_abs_path)}
    edit_config(str(to_edit), updates, DEBUG=True)

    subfolder = Path(abs_dir_name).joinpath('training_data')
    to_edit = subfolder.joinpath('training_data_config.yaml')
    new_abs_path = subfolder.joinpath('preprocessing_config.yaml')
    updates = {'preprocessing_config': str(new_abs_path)}
    edit_config(str(to_edit), updates)


#####################
# Filename utils
#####################

def get_project_name(_config):
    # Use current time
    project_name = datetime.now().strftime("%Y_%m_%d")
    exp = _config['experimenter']
    task = _config['task_name']
    project_name = f"{exp}-{task}-" + project_name
    return project_name


def get_sequential_filename(fname):
    """Check if the file or dir exists, and if so, append an integer"""
    i = 1
    if Path(fname).exists():
        print(f"Original fname {fname} exists, so will be suffixed")
        fname = fname + f"-{i}"
        while Path(fname).exists():
            i += 1
            fname = fname[:-2] + f"-{i}"
    return fname


def get_absname(project_path, fname):
    # Builds the absolute filepath using a project config filepath
    project_dir = Path(project_path).parent
    fname = Path(project_dir).joinpath(fname)
    return str(fname)

#####################
# config utils
#####################

def edit_config(config_fname, edits, DEBUG=False):
    """Generic overwriting, based on DLC"""

    if DEBUG:
        print(f"Editing config file at: {config_fname}")
    cfg = load_config(config_fname)
    if DEBUG:
        print(f"Initial config: {cfg} {type(cfg)}")
        print(f"Edits: {edits}")

    for k, v in edits.items():
        cfg[k] = v

    with open(config_fname, "w") as f:
        YAML().dump(cfg, f)

    return cfg


def load_config(config_fname):
    assert osp.exists(config_fname), f"{config_fname} not found!"

    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    return cfg

#####################
# Synchronizing config files
#####################

def synchronize_segment_config(project_path, segment_cfg):
    # For now, does NOT overwrite anything on disk
    project_cfg = load_config(project_path)

    updates = {'video_path': project_cfg['red_bigtiff_fname']}
    segment_cfg.update(updates)

    segment_folder = get_absname(project_path, 'segmentation')
    updates = {'output_folder': segment_folder}
    segment_cfg['output_params'].update(updates)

    return segment_cfg
