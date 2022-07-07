import argparse

from wbfm.gui.utils.napari_trace_explorer import napari_trace_explorer_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    DEBUG = args.DEBUG

    print("Starting trace explorer GUI, may take up to a minute to load...")

    napari_trace_explorer_from_config(project_path)
