import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.utils.general.utils_matplotlib import corrfunc, paired_boxplot_from_dataframes
from wbfm.utils.projects.finished_project_data import load_all_projects_from_list, ProjectData
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding, NeuronEncodingBase


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
                    logging.warning(f"Skipping invalid project {p.project_data.shortened_name}")
                    continue
                out = getattr(p, item)(*args, **kwargs)
                output[p.project_data.shortened_name] = out
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
