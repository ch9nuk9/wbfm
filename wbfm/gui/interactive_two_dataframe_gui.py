import argparse
import os
import pickle
import pandas as pd

from wbfm.gui.utils.utils_dash import dashboard_from_two_dataframes


def main(folder_name: str, port: int = 8070, allow_public_access: bool = False, DEBUG: bool = False):

    # Load the data from the folder.
    # There should be two files: a .h5 file for df_summary, and a .pickle file for the raw_dfs
    for f in os.listdir(folder_name):
        if f.startswith('.') or os.path.isdir(f):
            continue
        if f.endswith('.h5'):
            df_summary = pd.read_hdf(os.path.join(folder_name, f))
        elif f.endswith('.pickle'):
            with open(os.path.join(folder_name, f), 'rb') as handle:
                raw_dfs = pickle.load(handle)

    app = dashboard_from_two_dataframes(df_summary, raw_dfs)
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
    parser.add_argument('--DEBUG', default=False, help='')
    args = parser.parse_args()
    project_path = args.project_path
    port = args.port
    allow_public_access = args.allow_public_access
    allow_public_access = True if allow_public_access == "True" else False
    DEBUG = args.DEBUG

    main(project_path, port, allow_public_access, DEBUG)
