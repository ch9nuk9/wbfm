from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from tsnecuda import TSNE
from hdbscan import HDBSCAN
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

    n_clusters_per_window: int = 10
    n_volumes_per_window: int = 100

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

        for i, label in enumerate(tqdm(db_svd.labels_)):
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
        opt_tsne = dict(n_components=2, perplexity=10, early_exaggeration=200, force_magnify_iters=500)
        opt_db = dict(min_cluster_size=60, min_samples=10, max_cluster_size=110, cluster_selection_method='leaf')

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
        df_base = None
        all_dfs = []
        all_tsnes = []
        for _ in tqdm(range(num_clusters), leave=False):
            db_svd, Y_tsne_svd = self.cluster_single_window(start_volume)
            df = self.cluster_obj2dataframe(db_svd, start_volume)

            if df_base is None:
                df_base = df
            else:
                df = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)

            all_dfs.append(df)
            all_tsnes.append(Y_tsne_svd)  # TODO: check kl divergence of tsne?

        # Combine to one dataframe
        df_combined = combine_dataframes_using_mode(all_dfs)
        return df_combined
