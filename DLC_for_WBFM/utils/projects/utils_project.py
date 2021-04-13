import os.path as osp
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from ruamel.yaml import YAML


def build_project_structure(_config):

    # parent_folder = Path(_config['project_dir']).resolve()
    parent_folder = _config['project_dir']
    rel_dir_name = get_project_name(_config)

    # Build blank folder structure
    abs_dir_name = osp.join(parent_folder, rel_dir_name)
    abs_dir_name = get_sequential_filename(abs_dir_name)
    print(f"Building new project at: {abs_dir_name}")
    Path(abs_dir_name).mkdir(parents=True)

    subfolders = list(_config['subfolder_configs'])
    for sub in subfolders:
        Path(osp.join(abs_dir_name, sub)).mkdir(parents=True)

    # Copy config files and readmes over
    tmp = list(_config['subfolder_configs'].items())
    for subfolder, src in tmp:
        # Config file
        dest_fname = osp.basename(src)
        dest = osp.join(abs_dir_name, subfolder, dest_fname)
        copyfile(src, dest)
        # Also update in the config file itself
        _config['subfolder_configs'][subfolder] = osp.join(subfolder, dest_fname)
        # README
        dest_fname = 'README.md'
        dest = osp.join(abs_dir_name, subfolder, dest_fname)
        src_readme = osp.join(osp.dirname(src), 'README.md')
        copyfile(src_readme, dest)

    # Also copy those outside the subfolders
    src = _config['preprocessing_config']
    dest_fname = 'preprocessing_config.yaml'
    dest = osp.join(abs_dir_name, dest_fname)
    copyfile(src, dest)
    _config['preprocessing_config'] = dest_fname

    # the current project_config file itself
    # TODO: hardcoded
    dest_fname = 'project_config.yaml'
    src = osp.join('new_project_defaults', dest_fname)
    project_fname = osp.join(abs_dir_name, dest_fname)
    copyfile(src, project_fname)

    # Finally, update the copied project config with the new dest folder
    project_fname = str(Path(project_fname).resolve())
    edit_config(project_fname, _config)


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


def edit_config(config_fname, edits, DEBUG=False):
    """Generic overwriting, based on DLC"""

    assert osp.exists(config_fname), f"{config_fname} not found!"

    if DEBUG:
        print(f"Editing config file at: {config_fname}")
    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)
    if DEBUG:
        print(f"Initial config: {cfg} {type(cfg)}")
        print(f"Edits: {edits}")

    for k, v in edits.items():
        cfg[k] = v

    with open(config_fname, "w") as f:
        YAML().dump(cfg, f)

    return cfg
