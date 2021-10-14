import argparse

from DLC_for_WBFM.gui.utils.manual_annotation import create_manual_correction_gui
from DLC_for_WBFM.gui.utils.napari_match_correction import build_napari_match_corrector
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    parser.add_argument('--corrector_name', default=None,
                        help='name of the person doing the correction')
    parser.add_argument('--matches', default=None,
                        help='path to a previous or partial annotation to use as the starting point')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    corrector_name = args.corrector_name
    initial_annotation_name = args.matches
    DEBUG = args.DEBUG

    print("Starting manual annotation GUI, may take a while to load...")

    build_napari_match_corrector(project_path, initial_annotation_name, DEBUG=DEBUG)
