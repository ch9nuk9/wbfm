import argparse
import os
import pandas as pd

from wbfm.gui.utils.utils_dash import dashboard_from_two_dataframes
from wbfm.utils.general.utils_filenames import pickle_load_binary


def main(folder_name: str, port: int = None, allow_public_access: bool = False, DEBUG: bool = False,
         **kwargs):
    if port is None:
        port = 8070

    # Load the data from the folder.
    # There should be two files: a .h5 file for df_summary, and a .pickle file for the raw_dfs
    for f in os.listdir(folder_name):
        if f.startswith('.') or os.path.isdir(f):
            continue
        fname = os.path.join(folder_name, f)
        if f.endswith('.h5'):
            df_summary = pd.read_hdf(fname)
        elif f.endswith('.pickle'):
            raw_dfs = pickle_load_binary(fname)

    app = dashboard_from_two_dataframes(df_summary, raw_dfs, **kwargs)
    if allow_public_access:
        app.run_server(debug=DEBUG, port=port, host="0.0.0.0")
    else:
        app.run_server(debug=DEBUG, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--folder_path', '-p', default=None,
                        help='path to folder with .h5 and .pickle files')
    parser.add_argument('--allow_public_access', default=False,
                        help='allow access using the intranet (NOT SECURE)')
    parser.add_argument('--port', default=None,
                        help='port')
    parser.add_argument('-x', default=None, help='Default x axis for scatter plot')
    parser.add_argument('-y', default=None, help='Default y axis for scatter plot')
    parser.add_argument('-c', default=None, help='Default color splitting for scatter plot')
    parser.add_argument('--DEBUG', default=False, help='')

    args = parser.parse_args()
    folder_path = args.folder_path
    port = args.port
    allow_public_access = args.allow_public_access
    allow_public_access = True if allow_public_access == "True" else False
    x_default = args.x
    y_default = args.y
    c_default = args.c
    DEBUG = args.DEBUG

    main(folder_path, port, allow_public_access,
         x_default=x_default, y_default=y_default, color_default=c_default, DEBUG=DEBUG)
