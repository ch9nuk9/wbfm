import argparse
from wbfm.gui.utils.napari_trace_explorer import napari_behavior_explorer_from_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build behavior GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--fluorescence_fps', '-f', default=False,
                        help='fluorescence_fps')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    # Parse the string to determine if it is a boolean
    fluorescence_fps = args.fluorescence_fps
    if fluorescence_fps == 'False':
        fluorescence_fps = False
    elif fluorescence_fps == 'True':
        fluorescence_fps = True
    else:
        fluorescence_fps = True
    DEBUG = args.DEBUG

    print("Starting behavior explorer GUI, may take up to a minute to load...")

    napari_behavior_explorer_from_config(project_path, fluorescence_fps=fluorescence_fps, DEBUG=DEBUG)
