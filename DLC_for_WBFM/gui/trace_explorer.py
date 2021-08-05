import argparse

from DLC_for_WBFM.gui.utils.napari_trace_explorer import build_napari_trace_explorer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    DEBUG = args.DEBUG

    print("Starting trace explorer GUI, may take a while to load...")

    build_napari_trace_explorer(project_path)