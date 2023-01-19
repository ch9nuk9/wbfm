import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler


def calculate_residual_subtract_pca(df: pd.DataFrame, n_components=2) -> pd.DataFrame:
    """Note: must not contain nan"""
    pca = PCA(n_components=n_components, whiten=False)

    df_dat = df.to_numpy()
    scaler = StandardScaler(with_std=False)
    scaler.fit(df_dat)
    dat_normalized = scaler.transform(df_dat)
    dat_low_dimensional = pca.fit_transform(dat_normalized)
    dat_reprojected = pca.inverse_transform(dat_low_dimensional)
    dat_imputed = scaler.inverse_transform(dat_reprojected)

    dat_residual = df_dat - dat_imputed
    df_residual = pd.DataFrame(data=dat_residual, columns=df.columns)

    return df_residual


def calculate_residual_subtract_nmf(df: pd.DataFrame, n_components=2) -> pd.DataFrame:
    """Note: must not contain nan"""
    model = NMF(n_components=n_components)

    dat_normalized = df.to_numpy()
    dat_normalized += dat_normalized.min()
    dat_low_dimensional = model.fit_transform(dat_normalized)
    dat_imputed = model.inverse_transform(dat_low_dimensional)

    dat_residual = df - dat_imputed
    df_residual = pd.DataFrame(data=dat_residual, columns=df.columns)

    return df_residual
