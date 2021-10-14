import os.path as osp
from pathlib import Path
from sacred.observers import TinyDbReader

from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, get_project_of_substep


def _load_cfg(project_path):
    cfg = load_config(project_path)
    cfg = cfg
    project_dir = Path(project_path).parent
    return cfg, project_dir


def check_segmentation(project_path):
    cfg, project_dir = _load_cfg(project_path)

    try:
        with safe_cd(project_dir):
            segment_fname = cfg['subfolder_configs']['segmentation']
            this_cfg = load_config(segment_fname)
            # Segmentation subfolder may be from a different project
            other_project = get_project_of_substep(segment_fname)
            with safe_cd(other_project):
                all_to_check = [
                    this_cfg['output_masks'],
                    this_cfg['output_metadata']
                ]
                all_exist = map(osp.exists, all_to_check)

                return all(all_exist)
    except AssertionError:
        return False


def check_training(project_path):
    cfg, project_dir = _load_cfg(project_path)

    try:
        with safe_cd(project_dir):
            training_folder = Path(cfg['subfolder_configs']['training_data']).parent
            file_names = ['clust_df_dat.pickle', 'frame_dat.pickle', 'match_dat.pickle']
            all_to_check = map(lambda file: osp.join(training_folder, 'raw', file), file_names)
            all_exist = map(osp.exists, all_to_check)

            return all(all_exist)
    except AssertionError:
        return False


def check_tracking(project_path):
    cfg, project_dir = _load_cfg(project_path)

    try:
        with safe_cd(project_dir):
            tracking_folder = Path(cfg['subfolder_configs']['tracking']).parent
            all_to_check = [osp.join(tracking_folder, 'full_3d_tracks.h5')]
            all_exist = map(osp.exists, all_to_check)

            return all(all_exist)
    except AssertionError:
        return False


def check_traces(project_path):
    cfg = ModularProjectConfig(project_path)

    try:
        with safe_cd(cfg.project_dir):
            traces_cfg = cfg.get_traces_config()
            file_names = ['all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            make_full_name = lambda file: traces_cfg.resolve_relative_path(file, prepend_subfolder=True)
            all_to_check = map(make_full_name, file_names)
            all_exist = map(osp.exists, all_to_check)

            return all(all_exist)
    except AssertionError:
        return False


def print_sacred_log(project_path):
    cfg = ModularProjectConfig(project_path)

    reader = TinyDbReader(cfg.get_log_dir())
    results = reader.fetch_report(indices=-1)

    try:
        print(results[0])
    except KeyError:
        print("Key error in the log; this means a step is in progress or the log is corrupted")
