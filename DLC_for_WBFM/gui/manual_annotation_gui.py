import argparse
from pathlib import Path

from DLC_for_WBFM.gui.utils.manual_annotation import create_manual_correction_gui
from DLC_for_WBFM.utils.projects.utils_project import safe_cd, load_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    parser.add_argument('--corrector_name', default=None,
                        help='name of the person doing the correction')
    parser.add_argument('--annotation', default=None,
                        help='path to a previous or partial annotation to use as the starting point')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    corrector_name = args.corrector_name
    initial_annotation_name = args.annotation
    DEBUG = args.DEBUG

    print("Starting manual annotation GUI, may take a while to load...")

    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        trace_fname = Path(project_cfg['subfolder_configs']['traces'])
        trace_cfg = dict(load_config(trace_fname))
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))
        seg_fname = Path(project_cfg['subfolder_configs']['segmentation'])
        segment_cfg = dict(load_config(seg_fname))

    this_config = {'track_cfg': track_cfg, 'segment_cfg': segment_cfg, 'project_cfg': project_cfg,
                   'dataset_params': project_cfg['dataset_params'].copy(),
                   'project_dir': project_dir}

    with safe_cd(project_dir):
        create_manual_correction_gui(this_config, corrector_name, initial_annotation_name, DEBUG=DEBUG)
