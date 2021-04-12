import os.path as osp
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from ruamel.yaml import YAML


def build_project_structure(_config, path):

    dir_name = get_project_name(_config)

    # Build blank folder structure
    Path(dir_name).mkdir(parents=True, exist_ok=False)

    subfolders = list(_config['subfolder_configs'])
    for sub in subfolders:
        Path(osp.join(dir_name, sub)).mkdir(parents=True)

    # Copy config files and readmes over
    tmp = list(_config['subfolder_configs'].items())
    for subfolder, src in tmp:
        # Config file
        dest = osp.basename(src)
        dest = osp.join(dir_name, subfolder, dest)
        copyfile(src, dest)
        # Also update in the config file itself
        _config['subfolder_configs'][subfolder] = dest
        # README
        dest = 'README.md'
        dest = osp.join(dir_name, subfolder, dest)
        src_readme = osp.join(osp.dirname(src), 'README.md')
        copyfile(src_readme, dest)

    # Also copy those outside the subfolders
    src = _config['preprocessing_config']
    dest = osp.join(dir_name, 'preprocessing_config.yaml')
    copyfile(src, dest)
    _config['preprocessing_config'] = dest

    # the current project_config file itself
    # TODO: hardcoded
    src = osp.join('new_project_defaults', 'project_config.yaml')
    project_fname = osp.join(dir_name, 'project_config.yaml')
    copyfile(src, project_fname)
    # _config['project_path'] = dest

    # Finally, update the copied project config with the new dest folder
    edit_config(project_fname, _config)


def get_project_name(video_path, _config):
    if _config['output_params']['output_folder'] is None:
        output_folder = osp.split(video_path)[0]
    else:
        output_folder = _config['output_params']['output_folder']

    # Make a subfolder
    subfolder = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")
    subfolder = osp.join(output_folder, subfolder)
    Path(subfolder).mkdir(parents=True, exist_ok=True)

    # Make a suffix
    num_frames = _config['dataset_params']['num_frames']

    # Actual masks
    fname = f'masks_{num_frames}.btf'
    mask_fname = osp.join(subfolder, fname)

    # Metadata
    fname = f'metadata_{num_frames}.pickle'
    metadata_fname = osp.join(subfolder, fname)

    return mask_fname, metadata_fname


def edit_config(config_fname, edits):
    """Generic overwriting, based on DLC"""

    yaml = YAML()
    cfg = yaml.load(config_fname)

    for k, v in edits:
        cfg[k] = v

    with open(config_fname, "w") as f:
        yaml.dump(cfg, f)

    return cfg
