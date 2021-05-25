import os.path as osp
from pathlib import Path

from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, get_subfolder, get_project_of_substep


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
                    this_cfg['output']['masks'],
                    this_cfg['output']['metadata']
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
    cfg, project_dir = _load_cfg(project_path)

    try:
        with safe_cd(project_dir):
            traces_folder = Path(cfg['subfolder_configs']['traces']).parent
            file_names = ['all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            all_to_check = map(lambda file: osp.join(traces_folder, file), file_names)
            all_exist = map(osp.exists, all_to_check)

            return all(all_exist)
    except AssertionError:
        return False
