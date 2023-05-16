import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import detrend
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import split_flattened_index
from wbfm.utils.general.postprocessing.position_postprocessing import impute_missing_values_in_dataframe
from wbfm.utils.general.utils_matplotlib import corrfunc, paired_boxplot_from_dataframes
from wbfm.utils.projects.finished_project_data import load_all_projects_from_list, ProjectData
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages, \
    clustered_triggered_averages_from_list_of_projects
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding, NeuronEncodingBase
from wbfm.utils.visualization.filtering_traces import remove_outliers_using_std


@dataclass
class MultiProjectWrapper:
    all_project_paths: list = None
    all_projects: Dict = None

    class_constructor: callable = None
    constructor_kwargs: dict = field(default_factory=dict)
    use_threading: bool = True

    _all_behavior_plotters: List = None

    def __post_init__(self):
        if self.all_projects is None:
            assert self.all_project_paths is not None, "Must pass either projects or paths"
            self.all_projects = load_all_projects_from_list(self.all_project_paths)
        # Initialize the behavior plotters
        self._all_behavior_plotters = [self.class_constructor(p, **self.constructor_kwargs) for
                                       p in self.all_projects.values()]

    def __getattr__(self, item):
        # Transform all unknown function calls into a loop of calls to the subobjects
        def method(*args, **kwargs):
            print(f"Dynamically dispatching method: {item}")
            if item == '_all_behavior_plotters':
                return self._all_behavior_plotters
            output = {}
            for p in tqdm(self._all_behavior_plotters):
                if not p.is_valid:
                    logging.warning(f"Skipping invalid project {p.shortened_name}")
                    continue
                out = getattr(p, item)(*args, **kwargs)
                output[p.shortened_name] = out
            return output
        return method

    @staticmethod
    def concat_multiple_datasets_long(dict_of_dfs, long_format=True):
        # Works for get_data_for_paired_boxplot
        if long_format:
            df = pd.concat(dict_of_dfs, axis=1)  # Creates a multiindex dataframe
        else:
            df = pd.concat(dict_of_dfs, axis=0).T
        df = df.T.reset_index().drop(columns='level_1')
        if 'dataset_name' in df:
            df.drop('level_0', inplace=True)
        else:
            df = df.rename(columns={'level_0': 'dataset_name'})
        return df.T

    def __repr__(self):
        return f"Multiproject analyzer with {len(self._all_behavior_plotters)} projects"

    def set_for_all_classes(self, updates: dict):
        for key, val in updates.items():
            for b in self._all_behavior_plotters:
                b.__setattr__(key, val)


class MultiProjectTriggeredAverages(MultiProjectWrapper):

    class_constructor: callable = FullDatasetTriggeredAverages


class MultiProjectBehaviorPlotter(MultiProjectWrapper):

    class_constructor: callable = NeuronToMultivariateEncoding

    def pairplot_multi_dataset(self, which_channel='red', include_corr=True,
                               to_save=False):
        """
        Plots a seaborn pairplot for multiple datasets

        Parameters
        ----------
        which_channel
        include_corr
        to_save

        Returns
        -------

        """

        all_dfs = self.calc_per_neuron_df(which_channel)

        df = pd.concat(all_dfs, axis=0)
        df = df.reset_index().rename(columns={'index': 'neuron_name'})
        g = sns.pairplot(df, hue='dataset_name')
        if include_corr:
            g.map_lower(corrfunc)

        return g

        # if to_save:
        #     fname = '/home/charles/Current_work/presentations/nov_2022'
        #     fname = os.path.join(fname, 'gcamp6f_red_summary.png')
        #     plt.savefig(fname)

    def paired_boxplot_per_neuron_multi_dataset(self, df_start_name='red', df_final_name='ratio'):
        """
        Designed for use with subclass: BehavioralEncoding
            Uses per-neuron dataframes from each dataset

        Parameters
        ----------
        df_start_name
        df_final_name

        Returns
        -------

        """
        all_dfs = self.get_data_for_paired_boxplot(df_final_name, df_start_name)
        df = self.concat_multiple_datasets_long(all_dfs)

        paired_boxplot_from_dataframes(df.iloc[1:, :], [df_start_name, df_final_name])
        plt.title("Maximum correlation to kymograph")
        plt.ylim(0, 0.8)

    def paired_boxplot_overall_multi_dataset(self, df_name='ratio', **kwargs):
        """
        Designed for use with subclass: SpeedEncoding
            Uses full-dataset dataframes from each dataset (one number per dataset)

        Parameters
        ----------
        df_name

        Returns
        -------

        """
        dict_of_dfs = self.calc_dataset_summary_df(df_name, **kwargs)
        df = pd.concat(dict_of_dfs, axis=0).reset_index(drop=True).T

        paired_boxplot_from_dataframes(df)
        if kwargs.get('y_train', None) is not None:
            plt.title(f"Decoding of {kwargs['y_train']}")
        else:
            plt.title(f"Decoding of Speed")
        plt.ylim(-1.0, 1.0)


def build_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData],
                                             behavior_names: Union[str, List[str]],
                                             z_score_beh=False) -> pd.DataFrame:
    """
    Builds a time series of behavior from multiple projects

    See calc_behavior_from_alias for valid values of behavior_name

    Parameters
    ----------
    all_projects
    behavior_names
    z_score_beh: bool (default False) - whether to z-score the behavior (per dataset)

    Returns
    -------

    """

    if isinstance(behavior_names, str):
        behavior_names = [behavior_names]

    list_of_beh_dfs = []
    for b in behavior_names:
        output_dict = defaultdict(list)
        for dataset_name, p in all_projects.items():
            worm = p.worm_posture_class
            trace = worm.calc_behavior_from_alias(b)
            output_dict[b].extend(trace)
            output_dict['dataset_name'].extend([dataset_name] * len(trace))
            output_dict['local_time_index'].extend(np.arange(len(trace)))
        # Make sure the final dataframe is sorted correctly
        df_beh = pd.DataFrame(output_dict)
        df_beh = df_beh.sort_values(['dataset_name', 'local_time_index']).reset_index(drop=True)
        if z_score_beh:
            dataset_names_column = df_beh['dataset_name']
            df_beh = df_beh.groupby('dataset_name', group_keys=False).apply(lambda x: (x - x.mean(numeric_only=True)) / x.std(numeric_only=True))
            df_beh['dataset_name'] = dataset_names_column
        list_of_beh_dfs.append(df_beh)
    # Combine all the dataframes, keeping only a single column of dataset names
    df_beh = pd.concat(list_of_beh_dfs, axis=1)
    df_beh = df_beh.loc[:, ~df_beh.columns.duplicated()]
    return df_beh


def build_time_series_from_multiple_project_clusters(all_projects: Dict[str, ProjectData],
                                                     cluster_opt: dict = None,
                                                     z_score=False, trigger_opt: dict = None,
                                                     num_clusters=10,
                                                     **kwargs):
    """
    Similar to build_time_series_from_multiple_projects, using a clustering object to cluster traces across datasets.
    This object should be built from multiple datasets, and have names that can be split using split_flattened_index

    Note that this has all clusters as column names, and all concatenated time series as rows

    Parameters
    ----------
    all_projects
    z_score : bool - if True, z-scores the traces within each dataset (does not affect the clustering)

    Returns
    -------

    """
    if cluster_opt is None:
        cluster_opt = {'cluster_criterion': 'maxclust', 'linkage_threshold': num_clusters}
    # First build the clustering class
    multi_dataset_clusterer, clustering_intermediates = \
        clustered_triggered_averages_from_list_of_projects(all_projects,
                                                           cluster_opt=cluster_opt, trigger_opt=trigger_opt, **kwargs)
    all_triggered_average_classes, df_triggered_good, dict_of_triggered_traces = clustering_intermediates

    df_all_clusters, df_imputed = build_dataframe_of_clusters(all_triggered_average_classes, multi_dataset_clusterer,
                                                              z_score)

    return df_imputed, df_all_clusters, multi_dataset_clusterer, all_triggered_average_classes


def build_dataframe_of_clusters(all_triggered_average_classes, multi_dataset_clusterer, z_score: bool = False):
    """

    Parameters
    ----------
    all_triggered_average_classes
    multi_dataset_clusterer
    z_score

    Returns
    -------

    """
    # Loop through clusters, and do two things:
    # 1. Get the average of all neurons per dataset
    # 2. Save in a long dataframe
    # Get a dict from each dataset to the neurons within each cluster in that data
    clust_and_dataset_to_neurons = {}
    for i_clust, combined_names in tqdm(multi_dataset_clusterer.per_cluster_names.items()):

        dataset_to_neurons = defaultdict(list)
        for name in combined_names:
            dataset_name, neuron_name = split_flattened_index([name])[name]
            dataset_to_neurons[dataset_name].append(neuron_name)
        clust_and_dataset_to_neurons[i_clust] = dataset_to_neurons
    list_of_dfs = []
    for i_clust, dataset_to_neurons in tqdm(clust_and_dataset_to_neurons.items()):
        one_clust_dict = defaultdict(list)
        for dataset_name, these_neurons in dataset_to_neurons.items():
            # Each trace is an average of multiple time series (neurons)
            this_class = all_triggered_average_classes[dataset_name]

            traces = this_class.df_traces[these_neurons]
            trace_mean = traces.mean(axis=1)
            if z_score:
                trace_mean = trace_mean - trace_mean.mean()
                trace_mean = trace_mean / trace_mean.std()

            one_clust_dict[f'cluster_{i_clust}'].extend(list(trace_mean))
            one_clust_dict['dataset_name'].extend([dataset_name] * len(trace_mean))

        list_of_dfs.append(pd.DataFrame(one_clust_dict))
    # Make a temp index to keep things in order within each dataframe
    for df in list_of_dfs:
        df['local_time_index'] = df.groupby('dataset_name').cumcount()
    # Merge entire list of dataframes on the dataset name and the new index
    df_all_clusters = reduce(
        lambda left, right: pd.merge(left, right, on=['dataset_name', 'local_time_index'], how='outer'),
        list_of_dfs)
    df_all_clusters = df_all_clusters.sort_values(['dataset_name', 'local_time_index']).reset_index(drop=True)
    # Remove missing values to make valid for sklearn
    try:
        df_imputed = df_all_clusters.dropna(axis=1, thresh=df_all_clusters.shape[0] * 0.9).copy()
        df_imputed.drop(columns=['dataset_name', 'local_time_index'], inplace=True)
        df_imputed = remove_outliers_using_std(df_imputed, std_factor=4)
        df_imputed = impute_missing_values_in_dataframe(df_imputed)
    except ValueError as e:
        df_imputed = None
        logging.warning("Could not impute missing values in dataframe, returning None")
        print(e)
    return df_all_clusters, df_imputed


def get_variance_explained(project_data, trace_kwargs=None):
    if trace_kwargs is None:
        trace_kwargs = dict(channel_mode='dr_over_r_20', min_nonnan=0.9, filter_mode='rolling_mean')
    X = project_data.calc_default_traces(**trace_kwargs, interpolate_nan=True)
    X = detrend(X, axis=0)
    pca = PCA(n_components=20, whiten=False)
    pca.fit(X.T)

    return pca.explained_variance_ratio_


def get_all_variance_explained(all_projects_gcamp, all_projects_gfp, all_projects_immob):
    # PERCENT VARIANCE EXPLAINED
    gcamp_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_gcamp.items())}
    gfp_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_gfp)}
    immob_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_immob)}
    # Cumulative sum
    gcamp_var_sum = np.array([np.cumsum(p) for p in gcamp_var.values()]).T
    gfp_var_sum = np.array([np.cumsum(p) for p in gfp_var.values()]).T
    immob_var_sum = np.array([np.cumsum(p) for p in immob_var.values()]).T

    return gcamp_var, gfp_var, immob_var, gcamp_var_sum, gfp_var_sum, immob_var_sum
