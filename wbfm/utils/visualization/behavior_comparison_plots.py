import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.gui.utils.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.projects.finished_project_data import ProjectData


@dataclass
class BehaviorPlotter:
    # TODO: refactor dataframes as dict
    # TODO: refactor loaded dataframes as argument
    # TODO: parent class for multiple datasets
    project_path: str

    use_ransac: bool = False
    project_data: ProjectData = None

    def __post_init__(self):
        project_data = ProjectData.load_final_project_data_from_config(self.project_path)
        self.project_data = project_data

        if not self.project_data.worm_posture_class.has_beh_annotation:
            logging.warning("Behavior annotation not found, this class will not work")

    @cached_property
    def _all_dfs(self) -> list:
        print("First time calculating traces, may take a while...")

        opt = dict(channel_mode='red')
        red = self.project_data.calc_default_traces(**opt)
        opt = dict(channel_mode='green')
        green = self.project_data.calc_default_traces(**opt)
        opt = dict(channel_mode='ratio')
        ratio = self.project_data.calc_default_traces(**opt)
        opt = dict(channel_mode='ratio_rolling_ransac')
        ratio_ransac = self.project_data.calc_default_traces(**opt)
        opt = dict(channel_mode='ratio', filter_mode='rolling_mean')
        ratio_filt = self.project_data.calc_default_traces(**opt)
        # Align columns
        to_drop = set(ratio_filt.columns) - set(red.columns)
        ratio_filt.drop(columns=to_drop, inplace=True)

        return [red, green, ratio, ratio_filt, ratio_ransac]

    @property
    def all_dfs(self):
        if self.use_ransac:
            return self._all_dfs
        else:
            return self._all_dfs[:-1]

    @cached_property
    def all_dfs_corr(self):
        kymo = self.project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)
        kymo_smaller = kymo.loc[:, 3:30].copy()

        all_dfs_corr = [np.abs(pd.concat([df, kymo_smaller], axis=1).corr()) for df in self.all_dfs]

        # Only get the corner we care about: traces vs. kymo
        ind_nonneuron = np.arange(self.all_dfs[0].shape[1], all_dfs_corr[0].shape[1])
        ind_neurons = np.arange(0, self.all_dfs[0].shape[1])
        all_dfs_corr = [df.iloc[ind_neurons, ind_nonneuron] for df in all_dfs_corr]
        return all_dfs_corr

    @property
    def all_labels(self):
        if self.use_ransac:
            return ['red', 'green', 'ratio', 'ratio_filt', 'ratio_ransac']
        else:
            return ['red', 'green', 'ratio', 'ratio_filt']

    @property
    def all_colors(self):
        if self.use_ransac:
            return ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
        else:
            return ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

    def calc_summary_df(self, name):
        """

        Parameters
        ----------
        name - str, one of self.all_labels

        Returns
        -------

        """
        df_corr = self.all_dfs_corr[self.all_labels.index(name)]
        df_traces = self.all_dfs[self.all_labels.index(name)]

        body_segment_argmax = df_corr.columns[df_corr.apply(pd.Series.argmax, axis=1)]
        body_segment_argmax = pd.Series(body_segment_argmax, index=df_corr.index)

        corr_max = df_corr.max(axis=1)
        median = df_traces.median(axis=0)
        var = df_traces.var(axis=0)

        df_all = pd.concat([median, var, body_segment_argmax, corr_max], axis=1)
        df_all.columns = ['median_brightness', 'var_brightness', 'body_segment_argmax', 'corr_max']

        # Add column with name of dataset
        df_all['dataset_name'] = self.project_data.shortened_name
        df_all.dataset_name = df_all.dataset_name.astype('category')

        return df_all

    def calc_wide_pairwise_summary_df(self, start_name, final_name, to_add_columns=True):
        """
        Calculates basic parameters for single data types, as well as phase shifts

        Returns a widened dataframe, with new columns for each variable

        Example usage (with seaborn):
            df = plotter_gcamp.calc_wide_pairwise_summary_df('red', 'green')
            sns.pairplot(df)

        Parameters
        ----------
        start_name
        final_name

        Returns
        -------

        """
        # Get data for both individually
        df_start = self.calc_summary_df(start_name)
        df_final = self.calc_summary_df(final_name)
        df = df_start.join(df_final, lsuffix=f"_{start_name}", rsuffix=f"_{final_name}")

        # Build additional numeric columns
        if to_add_columns:
            to_subtract = 'body_segment_argmax'
            df['phase_difference'] = df[f"{to_subtract}_{final_name}"] - df[f"{to_subtract}_{start_name}"]

        return df

    def calc_long_pairwise_summary_df(self, start_name, final_name):
        """
        Calculates basic parameters for single data types

        Returns a long dataframe, with new columns for the original datatype ('source_data')
        Is also reindexed, with a new column referring to neuron names (these are duplicated)

        Example usage:
            df = plotter_gcamp.calc_long_pairwise_summary_df('red', 'green')
            sns.pairplot(df, hue='source_data', palette={'red': 'pink', 'green': 'green'})

        Parameters
        ----------
        start_name
        final_name

        Returns
        -------

        """
        # Get data for both individually
        df_start = self.calc_summary_df(start_name)
        df_final = self.calc_summary_df(final_name)

        # Build columns and join
        df_start['source_data'] = start_name
        df_final['source_data'] = final_name

        df = pd.concat([df_start, df_final], axis=0)
        df.source_data = df.source_data.astype('category')
        df = df.reset_index().rename(columns={'index': 'neuron_name'})

        return df

    def plot_correlation_of_examples(self, **kwargs):
        # Calculate correlation dataframes
        self._multi_plot(self.all_dfs, self.all_dfs_corr, self.all_labels, self.all_colors,
                         project_data=self.project_data, **kwargs)

    def plot_correlation_histograms(self):
        all_max_corrs = [df_corr.max(axis=1) for df_corr in self.all_dfs_corr]

        plt.figure(dpi=100)
        plt.hist(all_max_corrs,
                 color=self.all_colors,
                 label=self.all_labels)
        plt.xlim(-0.2, 1)
        plt.title(self.project_data.shortened_name)
        plt.xlabel("Maximum correlation")
        plt.legend()

    def plot_histogram_difference_after_ratio(self, df_start_names=None, df_final_name='ratio'):
        if df_start_names is None:
            df_start_names = ['red', 'green']
        # Get data
        all_df_starts = [self.all_dfs_corr[self.all_labels.index(name)] for name in df_start_names]
        df_final = self.all_dfs_corr[self.all_labels.index(df_final_name)]

        # Get differences
        df_final_maxes = df_final.max(axis=1)
        all_diffs = [df_final_maxes - df.max(axis=1) for df in all_df_starts]

        # Plot
        plt.hist(all_diffs)

        plt.xlabel("Maximum correlation difference")
        plt.title(f"Correlation difference between {df_start_names} to {df_final_name}")

    def plot_paired_boxplot_difference_after_ratio(self, df_start_name='red', df_final_name='ratio'):
        # Get data
        df_start = self.all_dfs_corr[self.all_labels.index(df_start_name)]
        df_final = self.all_dfs_corr[self.all_labels.index(df_final_name)]

        start_maxes = df_start.max(axis=1)
        final_max = df_final.max(axis=1)
        both_maxes = pd.concat([start_maxes, final_max], axis=1).T

        # Plot
        paired_boxplot_from_dataframes(both_maxes, [df_final_name, df_start_name])

        plt.ylim(0, 0.8)
        plt.ylabel("Absolute correlation")
        plt.title("Change in body segment correlation")

    def plot_phase_difference(self, df_start_name='red', df_final_name='green', corr_thresh=0.2, remove_zeros=True):
        """
        Green minus red

        Returns
        -------

        """
        df_start = self.all_dfs_corr[self.all_labels.index(df_start_name)].copy()
        df_final = self.all_dfs_corr[self.all_labels.index(df_final_name)].copy()

        ind_to_keep = df_start.max(axis=1) > corr_thresh
        df_start = df_start.loc[ind_to_keep, :]
        df_final = df_final.loc[ind_to_keep, :]

        start_body_segment_argmax = df_start.columns[df_start.apply(pd.Series.argmax, axis=1)]
        final_body_segment_argmax = df_final.columns[df_final.apply(pd.Series.argmax, axis=1)]

        diff = final_body_segment_argmax - start_body_segment_argmax
        title_str = f"{df_final_name} - {df_start_name} with starting corr > {corr_thresh}"
        if remove_zeros:
            diff = diff[diff != 0]
            title_str = f"{title_str} (zeros removed)"
        plt.hist(diff, bins=np.arange(diff.min(), diff.max()))
        plt.title(title_str)
        plt.xlabel("Phase shift (body segments)")

    @staticmethod
    def _multi_plot(all_dfs, all_dfs_corr, all_labels, all_colors, ax_locations=None,
                    project_data: ProjectData=None,
                    corr_thresh=0.3, which_df_to_apply_corr_thresh=0, max_num_plots=None,
                    xlim=None, to_save=False):
        if xlim is None:
            xlim = [100, 450]
        if ax_locations is None:
            ax_locations = [1, 1, 3, 3, 3]

        all_names = list(all_dfs_corr[0].index)
        num_open_plots = 0

        for i in range(all_dfs_corr[0].shape[0]):
            corr = np.abs(all_dfs_corr[which_df_to_apply_corr_thresh].iloc[i, :])
            if corr.max() < corr_thresh:
                continue
            else:
                num_open_plots += 1

            fig, axes = plt.subplots(ncols=2, nrows=2, dpi=100, figsize=(15, 5))
            axes = np.ravel(axes)
            neuron_name = all_names[i]

            for df, df_corr, lab, col, ax_loc in zip(all_dfs, all_dfs_corr, all_labels, all_colors, ax_locations):

                plt_opt = dict(label=lab, color=col)
                # Always put the correlation on ax 0
                corr = df_corr.iloc[i, :]
                axes[0].plot(corr, **plt_opt)

                # Put the trace on the passed axis
                corr = df[neuron_name]
                axes[ax_loc].plot(corr / corr.mean(), **plt_opt)

            axes[0].set_xlabel("Body segment")
            axes[0].set_ylabel("Correlation")
            axes[0].set_ylim(0, 0.8)
            axes[0].set_title(neuron_name)
            axes[0].legend()

            for ax in [axes[1], axes[3]]:
                ax.set_xlim(xlim[0], xlim[1])
                ax.legend()
                ax.set_xlabel("Time (frames)")
                ax.set_ylabel("Normalized amplitude")
                if project_data is not None:
                    project_data.shade_axis_using_behavior(ax)

            # axes[2].remove()
            axes[2].plot(all_dfs[0][neuron_name], all_dfs[1][neuron_name], '.')
            axes[2].set_xlabel("Red")
            axes[2].set_ylabel("Green")

            fig.tight_layout()

            if to_save:
                vis_cfg = project_data.project_config.get_visualization_config()
                fname = f'traces_kymo_correlation_{neuron_name}.png'
                fname = vis_cfg.resolve_relative_path(vis_cfg, prepend_subfolder=True)

                plt.savefig(fname)

            if max_num_plots is not None and num_open_plots >= max_num_plots:
                break


@dataclass
class MultiProjectBehaviorPlotter:
    all_project_paths: list

    _all_behavior_plotters: List[BehaviorPlotter] = None

    def __post_init__(self):
        # Just initialize the behavior plotters
        self._all_behavior_plotters = [BehaviorPlotter(p) for p in self.all_project_paths]

    def __getattr__(self, item):
        # Transform all unknown function calls into a loop of calls to the subobjects
        def method(*args, **kwargs):
            output = []
            for p in tqdm(self._all_behavior_plotters):
                out = getattr(p, item)(*args, **kwargs)
                output.append(out)
            return output
            # print("tried to handle unknown method " + item)
            # if args:
            #     print("it had arguments: " + str(args) + str(kwargs))
        return method
