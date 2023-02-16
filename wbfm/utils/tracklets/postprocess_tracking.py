import warnings
from dataclasses import dataclass
from typing import List
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import dask
import numpy as np
from numpy import ma
from sklearn.preprocessing import StandardScaler
from ppca import PPCA
from sklearn.metrics.pairwise import nan_euclidean_distances
from tqdm.auto import tqdm

from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


def normalize_3d(all_dist):

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        all_dist_mean = np.nanmean(all_dist, axis=0)
        all_dist_std = np.nanstd(all_dist, axis=0)
        all_dist_norm = (all_dist - all_dist_mean) / all_dist_std
    return all_dist_norm


@dataclass
class OutlierRemoval:
    names: List[str]

    d: int = 10
    learning_rate: float = 0.4

    verbose: int = 1

    all_zxy_3d_original: np.ndarray = None
    df_traces: pd.DataFrame = None

    _all_zxy_3d: np.ndarray = None
    _all_dist_flattened: np.ndarray = None
    _all_dist: np.ndarray = None
    _next_matrix_to_remove: np.ndarray = None
    _outlier_values: np.ndarray = None
    total_matrix_to_remove: np.ndarray = None

    @staticmethod
    def load_from_project(project_data, verbose=0, **kwargs):

        coords = ['z', 'x', 'y']
        names = project_data.well_tracked_neuron_names(min_nonnan=0.9)
        df_traces = project_data.calc_default_traces(channel_mode='ratio', min_nonnan=0.9)

        all_zxy = project_data.green_traces.loc[:, (names, coords)].copy()
        z_to_xy_ratio = project_data.physical_unit_conversion.z_to_xy_ratio
        all_zxy.loc[:, (slice(None), 'z')] = z_to_xy_ratio * all_zxy.loc[:, (slice(None), 'z')]

        obj = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs)
        return obj

    @staticmethod
    def load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs):
        all_zxy = all_zxy.sort_index(axis='columns')
        all_zxy_3d = all_zxy.to_numpy().reshape(all_zxy.shape[0], -1, len(coords))
        # Save
        obj = OutlierRemoval(names,
                             verbose=verbose,
                             df_traces=df_traces,
                             all_zxy_3d_original=all_zxy_3d.copy(),
                             _all_zxy_3d=all_zxy_3d,
                             **kwargs)
        obj.get_pairwise_distances()
        return obj

    @staticmethod
    def load_from_df(df_tracks, df_traces=None, verbose=0, **kwargs):

        coords = ['z', 'x', 'y']
        names = get_names_from_df(df_tracks)

        all_zxy = df_tracks.loc[:, (names, coords)].copy()
        obj = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs)
        return obj

    def get_pairwise_distances(self):
        all_zxy_3d = self._all_zxy_3d
        # Get all pairwise distances
        output = []
        for time_zxy in all_zxy_3d:
            output.append(dask.delayed(nan_euclidean_distances)(time_zxy))
        total = dask.delayed(np.stack)(output)
        all_dist = total.compute()
        all_dist_flattened = all_dist.reshape(all_dist.shape[0], -1)
        self._all_dist_flattened = all_dist_flattened
        self._all_dist = all_dist
        return all_dist, all_dist_flattened

    def calc_outlier_indices_using_ppca(self):
        d = self.d
        learning_rate = self.learning_rate
        verbose = self.verbose

        all_dist_flattened = self._all_dist_flattened
        all_dist = self._all_dist
        names = self.names

        if verbose > 0:
            print("Building low dimensional manifold...")
        scaler = StandardScaler()
        scaler.fit(all_dist_flattened)
        dat_normalized = scaler.transform(all_dist_flattened)

        ppca = PPCA()
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            ppca.fit(data=dat_normalized, d=d, tol=0.02, verbose=(verbose > 1))

        if verbose > 0:
            print("Project data to that manifold...")
        V = ma.masked_invalid(ppca.C)
        full_time_mode_weights = ma.dot(ma.masked_invalid(dat_normalized), V)
        full_time_reconstruction = ma.dot(full_time_mode_weights, V.T)
        dat_imputed_flattened = scaler.inverse_transform(full_time_reconstruction)
        all_dist_imputed = dat_imputed_flattened.reshape(all_dist.shape)

        # Calculate the sum of distances over all paired neurons
        # But normalize each pairwise distance by the std
        all_dist_imputed_norm = normalize_3d(all_dist_imputed)
        all_dist_norm = normalize_3d(all_dist)

        all_dist_diff = np.abs(all_dist_imputed_norm - all_dist_norm)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            dat_outliers = np.nanmedian(all_dist_diff, axis=2)

        # Get individual outliers, and decide which to remove in this iteration
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            all_thresholds = 3*np.nanstd(dat_outliers, axis=0)
        matrix_to_remove = np.zeros_like(dat_outliers, dtype=bool)
        for i, name in enumerate(names):
            # Take the top percentage of the time points over the threshold
            these_values = dat_outliers[:, i]
            num_outliers = np.sum(these_values - np.nanmean(these_values) > all_thresholds[i])
            if num_outliers == 0:
                continue
            num_to_remove = int(num_outliers*learning_rate)
            if num_to_remove == 0:
                num_to_remove = 1
            ind_sort = np.argsort(-these_values)
            ind_to_remove = ind_sort[:num_to_remove]

            matrix_to_remove[ind_to_remove, i] = True

            if name == 'neuron_060':
                print(num_outliers, num_to_remove)

        self._next_matrix_to_remove = matrix_to_remove
        self._outlier_values = dat_outliers
        if self.total_matrix_to_remove is None:
            self.total_matrix_to_remove = matrix_to_remove
        else:
            self.total_matrix_to_remove = self.total_matrix_to_remove | matrix_to_remove

        return matrix_to_remove

    def remove_outliers_from_zxy(self):
        matrix_to_remove = self._next_matrix_to_remove
        self._all_zxy_3d[matrix_to_remove, :] = np.nan

    def iteratively_remove_outliers_using_ppca(self, max_iter=5, DEBUG=False):
        # Do not assume it was set up initially; start from all_zxy_3d
        for i in tqdm(range(max_iter)):
            self.get_pairwise_distances()
            self.calc_outlier_indices_using_ppca()
            self.remove_outliers_from_zxy()

            if DEBUG:
                self.plot_outlier_values('neuron_060')

            print(f"Removed {np.sum(self._next_matrix_to_remove)} outliers "
                  f"(total={np.sum(self.total_matrix_to_remove)})")

    def plot_before_after(self, neuron_name, z_not_traces=True):
        i_trace = self.names.index(neuron_name)

        mask_remove = self.total_matrix_to_remove[:, i_trace]
        ind_remove = np.where(mask_remove)[0]

        if z_not_traces:
            y0 = self.all_zxy_3d_original[:, i_trace, 0]
            y1 = self._all_zxy_3d[:, i_trace, 0]
        else:
            y0 = self.df_traces[neuron_name]
            y1 = y0.copy()
            y1[mask_remove] = np.nan
        df = pd.DataFrame([y0, y1], index=['Original', 'Filtered']).T
        y_remove = y0[mask_remove]

        fig = px.line(df, title=f"Num removed = {len(ind_remove)}")
        # print(ind_remove)
        fig.add_trace(go.Scatter(x=ind_remove, y=y_remove, mode='markers'))
        fig.show()

    def plot_outlier_values(self, neuron_name):
        i_trace = self.names.index(neuron_name)

        trace = self._outlier_values[:, i_trace].copy()
        trace -= np.nanmean(trace)

        mask_remove = self.total_matrix_to_remove[:, i_trace]
        y_remove = trace[mask_remove]
        ind_remove = np.where(mask_remove)[0]

        fig = px.line(trace, title=f"Num removed = {len(ind_remove)}")
        # print(ind_remove)
        fig.add_hline(3 * np.nanstd(trace))
        fig.add_trace(go.Scatter(x=ind_remove, y=y_remove, mode='markers'))
        fig.show()
