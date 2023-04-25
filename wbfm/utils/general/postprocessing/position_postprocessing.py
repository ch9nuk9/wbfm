from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from wbfm.utils.projects.utils_project import safe_cd


def impute_missing_values_using_config(tracking_config, DEBUG=False):
    """
    Using gappy time series of the positions of all neurons, uses probabilistic PCA (PPCA) to impute the missing values


    Parameters
    ----------
    project_config

    Returns
    -------

    """

    # Unpack config
    project_dir = tracking_config['project_dir']
    df_fname = tracking_config['final_3d_tracks_df']

    #
    with safe_cd(project_dir):
        df = pd.read_hdf(df_fname)

    df_imputed = impute_missing_values_in_dataframe(df)

    # Save
    out_fname = Path(df_fname).with_name('postprocessing').joinpath('full_3d_tracks.h5')
    df_imputed.to_hdf(out_fname, 'df_with_missing')
    df_imputed.to_csv(Path(out_fname).with_suffix('.csv'))


def impute_missing_values_in_dataframe(df: pd.DataFrame, d=None) -> pd.DataFrame:
    """
    Given a dataframe with gaps, impute the missing values using PPCA

    Parameters
    ----------
    df
    d

    Returns
    -------

    """
    from ppca import PPCA

    # DLC uses zeros as "failed tracking"
    # Replace with nan and scale
    df.replace(0, np.NaN, inplace=True)
    df_dat = df.to_numpy()
    scaler = StandardScaler()
    scaler.fit(df_dat)
    dat_normalized = scaler.transform(df_dat)
    # Actually impute
    ppca = PPCA()
    ppca.fit(data=dat_normalized, d=d, verbose=False)
    dat_imputed = scaler.inverse_transform(ppca.data)
    df_imputed = pd.DataFrame(data=dat_imputed, columns=df.columns)
    return df_imputed
