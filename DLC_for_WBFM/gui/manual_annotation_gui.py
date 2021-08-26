import argparse

from DLC_for_WBFM.gui.utils.manual_annotation import create_manual_correction_gui
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import safe_cd

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

    cfg = modular_project_config(project_path)
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()
    training_cfg = cfg.get_training_config()
    tracking_cfg = cfg.get_tracking_config()

    with safe_cd(project_dir):
        create_manual_correction_gui(cfg,
                                     segment_cfg,
                                     training_cfg,
                                     tracking_cfg,
                                     corrector_name, initial_annotation_name, DEBUG=DEBUG)
