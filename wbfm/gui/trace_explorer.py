import argparse
import logging

from wbfm.gui.utils.napari_trace_explorer import napari_trace_explorer_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--force_tracklets_to_be_sparse', default=True)
    parser.add_argument('--load_tracklets', default=True)
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    force_tracklets_to_be_sparse = args.force_tracklets_to_be_sparse
    force_tracklets_to_be_sparse = (force_tracklets_to_be_sparse == "True")
    load_tracklets = args.load_tracklets
    load_tracklets = (load_tracklets == "True")
    if not force_tracklets_to_be_sparse:
        logging.warning("Tracklets will not be forced to be sparse. This may cause interactivity to crash.")
    DEBUG = args.DEBUG

    print("Starting trace explorer GUI, may take up to a minute to load...")

    napari_trace_explorer_from_config(project_path, load_tracklets=load_tracklets,
                                      force_tracklets_to_be_sparse=force_tracklets_to_be_sparse,
                                      DEBUG=DEBUG)
