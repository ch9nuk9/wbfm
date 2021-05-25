import os.path as osp
from datetime import datetime
from pathlib import Path
from shutil import copytree
from ruamel.yaml import YAML
from contextlib import contextmanager
import os
# import concurrent.futures


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


@contextmanager
def safe_cd(newdir):
    # https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory/24176022#24176022
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

#####################
# config utils
#####################


def edit_config(config_fname, edits, DEBUG=False):
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

    # segment_folder = get_absname(project_path, 'segmentation')
    # updates = {'output_folder': segment_folder}
    # segment_cfg['output_params'].update(updates)

    return segment_cfg


def synchronize_train_config(project_path, train_cfg):
    # For now, does NOT overwrite anything on disk
    project_cfg = load_config(project_path)

    # Previous step, which produced needed files
    segment_cfg = load_config(project_cfg['subfolder_configs']['segmentation'])
    # This step; to update
    train_cfg = load_config(project_cfg['subfolder_configs']['training_data'])

    # Add external detections
    external_detections = segment_cfg['output']['metadata']
    if not os.path.exists(external_detections):
        raise FileNotFoundError("Could not find external annotations")
    # external_detections = segment_cfg['output_params']['output_folder']
    # Assume the detections are named normally, i.e. starting with 'metadata'
    # for file in os.listdir(external_detections):
    #     if fnmatch.fnmatch(file, 'metadata*'):
    #         external_detections = osp.join(external_detections, file)
    #         break
    # else:
    #     raise FileNotFoundError("Could not find external annotations")

    updates = {'external_detections': external_detections}
    train_cfg['tracker_params'].update(updates)

    return train_cfg


def get_subfolder(project_path, subfolder):
    project_cfg = load_config(project_path)
    return Path(project_cfg['subfolder_configs'][subfolder]).parent


def get_project_of_substep(subfolder_path):
    return Path(Path(subfolder_path).parent).parent