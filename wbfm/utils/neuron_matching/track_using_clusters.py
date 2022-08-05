from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from tsnecuda import TSNE
from hdbscan import HDBSCAN

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching, \
    combine_dataframes_using_mode

import matplotlib

from wbfm.utils.projects.utils_neuron_names import int2name_neuron


def plot_clusters(db, Y, class_labels=True):
    plt.figure(figsize=(15, 15))

    labels = db.labels_
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    # colors = [plt.cm.Set1(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors = matplotlib.colors.ListedColormap(np.random.rand(256, 3)).colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = Y[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        if class_labels:
            plt.annotate(f'{k}', np.mean(xy, axis=0), fontsize=24)

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()


@dataclass
class WormTsneTracker:
    X_svd: np.array
    linear_ind_to_local: list

    opt_tsne: dict = None
    opt_db: dict = None

    n_clusters_per_window: int = 5
    n_volumes_per_window: int = 120
    tracker_stride: int = None

    verbose: int = 1

    def __post_init__(self):
        # TODO: parameter search of these options
        self.opt_tsne = dict(n_components=2, perplexity=10, early_exaggeration=200, force_magnify_iters=500)
        self.opt_db = dict(min_cluster_size=int(0.6*self.n_volumes_per_window),
                           min_samples=int(0.1*self.n_volumes_per_window),
                           max_cluster_size=int(1.1*self.n_volumes_per_window),
                           cluster_selection_method='leaf')

        self.tracker_stride = int(0.5 * self.n_volumes_per_window)

    @property
    def num_frames(self):
        return len(self.linear_ind_to_local)

    @property
    def all_start_volumes(self):
        all_start_volumes = list(np.arange(0, self.num_frames - self.n_volumes_per_window, step=self.tracker_stride))
        all_start_volumes.append(self.num_frames - self.n_volumes_per_window - 1)
        return all_start_volumes

    def cluster_obj2dataframe(self, db_svd, start_volume):
        # Associate cluster label ids to a (time, local ind) tuple
        # i.e. build a dict
        # Note: the dict key should be a tuple of (neuron_name, 'raw_neuron_ind_in_list'), because we want it to be a multilevel dataframe

        linear_ind_to_local = self.linear_ind_to_local
        n_vols = self.n_volumes_per_window

        cluster_dict = {}
        # Assume the labels are in sequential order
        current_time = start_volume
        current_global_ind = list(linear_ind_to_local[current_time].copy())
        current_local_ind = 0

        for i, label in enumerate(db_svd.labels_):
            global_ind = current_global_ind.pop(0)

            if label == -1:
                # Still want to pop above
                pass
            else:
                this_neuron_name = int2name_neuron(label + 1)
                key = (this_neuron_name, 'raw_neuron_ind_in_list')

                if key not in cluster_dict:
                    tmp = np.empty(n_vols + start_volume)
                    tmp[:] = np.nan
                    cluster_dict[key] = tmp

                if np.isnan(cluster_dict[key][current_time]):
                    cluster_dict[key][current_time] = current_local_ind
                else:
                    # TODO: For now, just ignore the second assignment
                    pass
                    # print(f"Multiple assignments found for {this_neuron_name} at t={current_time}")

            if len(current_global_ind) == 0:
                current_time += 1
                current_global_ind = list(linear_ind_to_local[current_time].copy())
                current_local_ind = 0
            else:
                current_local_ind += 1
        df_cluster = pd.DataFrame(cluster_dict)
        return df_cluster

    def cluster_single_window(self, start_volume=0):
        # Unpack
        X_svd = self.X_svd
        linear_ind_to_local = self.linear_ind_to_local
        n_vols = self.n_volumes_per_window

        # Options
        opt_tsne = self.opt_tsne
        opt_db = self.opt_db

        # Get this window of data
        vol_ind = np.arange(start_volume, start_volume + n_vols)
        linear_ind = np.hstack([linear_ind_to_local[i] for i in vol_ind])

        # tsne + cluster
        tsne = TSNE(**opt_tsne)
        Y_tsne_svd = tsne.fit_transform(X_svd[linear_ind, :])
        db_svd = HDBSCAN(**opt_db).fit(Y_tsne_svd)

        return db_svd, Y_tsne_svd

    def multicluster_single_window(self, start_volume=0):
        """
        Cluster one window n times, and then combine for consistency

        Parameters
        ----------
        start_volume

        Returns
        -------

        """
        num_clusters = self.n_clusters_per_window

        # Get all iterations
        all_raw_dfs = []
        all_tsnes = []
        for _ in tqdm(range(num_clusters), leave=False):
            db_svd, Y_tsne_svd = self.cluster_single_window(start_volume)
            df = self.cluster_obj2dataframe(db_svd, start_volume)
            all_raw_dfs.append(df)
            all_tsnes.append(Y_tsne_svd)  # TODO: check kl divergence of tsne?

        # Choose a base dataframe and rename all to that one
        # TODO: for now just choosing the one with the most neurons
        i_most = np.argmax([df.shape[1] for df in all_raw_dfs])
        df_base = all_raw_dfs[i_most]
        all_dfs = [df_base]
        for i, df in enumerate(all_raw_dfs):
            if i == i_most:
                continue
            df_renamed, *_ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)
            all_dfs.append(df_renamed)

        # Combine to one dataframe
        if len(all_dfs) > 1:
            df_combined = combine_dataframes_using_mode(all_dfs)
        else:
            df_combined = all_dfs[0]

        return df_combined, all_raw_dfs

    def track_using_overlapping_windows(self):
        """
        Clusters one window, then moves by self.tracker_stride, clusters again, and combines in sequence

        Returns
        -------

        """

        all_start_volumes = self.all_start_volumes
        if self.verbose >= 1:
            print(f"Starting clustering of {len(all_start_volumes)} windows of length {self.n_volumes_per_window}")

        # Track each window
        all_dfs = []
        for start_volume in tqdm(all_start_volumes):
            df_window, _ = self.multicluster_single_window(start_volume)
            all_dfs.append(df_window)

        # Make them all the right shape, then iteratively rename them
        if self.verbose >= 1:
            print(f"Combining all dataframes to common namespace")
        all_dfs = [fill_missing_indices_with_nan(df, expected_max_t=self.num_frames)[0] for df in all_dfs]
        df_base = all_dfs[0]
        all_dfs_renamed = [df_base]
        for df in tqdm(all_dfs[1:]):
            df_renamed, *_ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)
            all_dfs_renamed.append(df_renamed)

        # Finally, combine
        df_combined = combine_dataframes_using_mode(all_dfs_renamed)

        # Reweight confidence

        return df_combined, all_dfs