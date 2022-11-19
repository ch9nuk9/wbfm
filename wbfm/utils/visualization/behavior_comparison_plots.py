import logging
from dataclasses import dataclass
from functools import reduce
from typing import List, Dict

import numpy as np
import pandas as pd
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from wbfm.gui.utils.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.projects.finished_project_data import ProjectData
import statsmodels.api as sm
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


@dataclass
class BehaviorPlotter:
    project_path: str

    dataframes_to_load: List[str] = None
    project_data: ProjectData = None

    def __post_init__(self):
        project_data = ProjectData.load_final_project_data_from_config(self.project_path)
        self.project_data = project_data

        if not self.project_data.worm_posture_class.has_beh_annotation:
            logging.warning("Behavior annotation not found, this class will not work")

        if self.dataframes_to_load is None:
            self.dataframes_to_load = ['red', 'green', 'ratio', 'ratio_filt']

    @cached_property
    def all_dfs(self) -> Dict[str, pd.DataFrame]:
        print("First time calculating traces, may take a while...")

        all_dfs = dict()
        for key in self.dataframes_to_load:
            # Assumes keys are a basic data mode, perhaps with a _filt suffix
            opt = dict()
            channel_key = key
            if '_filt' in key:
                channel_key = key.replace('_filt', '')
                opt['filter_mode'] = 'rolling_mean'
            opt['channel_mode'] = channel_key
            all_dfs[key] = self.project_data.calc_default_traces(**opt)

        # Align columns to commmon subset
        all_column_names = [df.columns for df in all_dfs.values()]
        common_column_names = reduce(np.intersect1d, all_column_names)
        all_to_drop = [set(df.columns) - set(common_column_names) for df in all_dfs.values()]
        for key, to_drop in zip(all_dfs.keys(), all_to_drop):
            all_dfs[key].drop(columns=to_drop, inplace=True)

        return all_dfs

    @cached_property
    def all_dfs_corr(self) -> Dict[str, pd.DataFrame]:
        kymo = self.project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)
        kymo_smaller = kymo.loc[:, 3:30].copy()

        all_dfs_corr = {key: np.abs(pd.concat([df, kymo_smaller], axis=1).corr()) for key, df in self.all_dfs.items()}

        # Only get the corner we care about: traces vs. kymo
        label0 = self.all_labels[0]
        ind_nonneuron = np.arange(self.all_dfs[label0].shape[1], all_dfs_corr[label0].shape[1])
        ind_neurons = np.arange(0, self.all_dfs[label0].shape[1])
        all_dfs_corr = {key: df.iloc[ind_neurons, ind_nonneuron] for key, df in all_dfs_corr.items()}
        return all_dfs_corr

    @property
    def all_labels(self):
        return list(self.all_dfs.keys())

    @property
    def all_colors(self):
        cols = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
        return cols[:len(self.all_labels)]

    def calc_summary_df(self, name):
        """

        Parameters
        ----------
        name - str, one of self.all_labels

        Returns
        -------

        """
        df_corr = self.all_dfs_corr[name]
        df_traces = self.all_dfs[name]

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
        self._multi_plot(list(self.all_dfs.values()), list(self.all_dfs_corr.values()),
                         self.all_labels, self.all_colors,
                         project_data=self.project_data, **kwargs)

    def plot_correlation_histograms(self):
        all_max_corrs = [df_corr.max(axis=1) for df_corr in self.all_dfs_corr.values()]

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
        all_df_starts = [self.all_dfs_corr[name] for name in df_start_names]
        df_final = self.all_dfs_corr[df_final_name]

        # Get differences
        df_final_maxes = df_final.max(axis=1)
        all_diffs = [df_final_maxes - df.max(axis=1) for df in all_df_starts]

        # Plot
        plt.hist(all_diffs)

        plt.xlabel("Maximum correlation difference")
        plt.title(f"Correlation difference between {df_start_names} to {df_final_name}")

    def plot_paired_boxplot_difference_after_ratio(self, df_start_name='red', df_final_name='ratio'):
        # Get data
        df_start = self.all_dfs_corr[df_start_name]
        df_final = self.all_dfs_corr[df_final_name]

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
        df_start = self.all_dfs_corr[df_start_name].copy()
        df_final = self.all_dfs_corr[df_final_name].copy()

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
    def _multi_plot(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations=None,
                    project_data: ProjectData=None,
                    corr_thresh=0.3, which_df_to_apply_corr_thresh=0, max_num_plots=None,
                    xlim=None, to_save=False):
        if xlim is None:
            xlim = [100, 450]
        if ax_locations is None:
            ax_locations = [1, 1, 3, 3, 3]

        # label0 = list(all_dfs_corr_list.keys())[0]
        all_names = list(all_dfs_corr_list[0].index)
        num_open_plots = 0

        for i in range(all_dfs_corr_list[0].shape[0]):
            corr = np.abs(all_dfs_corr_list[which_df_to_apply_corr_thresh].iloc[i, :])
            if corr.max() < corr_thresh:
                continue
            else:
                num_open_plots += 1

            fig, axes = plt.subplots(ncols=2, nrows=2, dpi=100, figsize=(15, 5))
            axes = np.ravel(axes)
            neuron_name = all_names[i]

            for df, df_corr, lab, col, ax_loc in zip(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations):

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

            axes[2].plot(all_dfs_list[0][neuron_name], all_dfs_list[1][neuron_name], '.')
            axes[2].set_xlabel("Red")
            axes[2].set_ylabel("Green")

            fig.tight_layout()

            if to_save:
                vis_cfg = project_data.project_config.get_visualization_config()
                fname = f'traces_kymo_correlation_{neuron_name}.png'
                fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

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
            output = {}
            for p in tqdm(self._all_behavior_plotters):
                out = getattr(p, item)(*args, **kwargs)
                output[p.project_data.shortened_name] = out
            return output
            # print("tried to handle unknown method " + item)
            # if args:
            #     print("it had arguments: " + str(args) + str(kwargs))
        return method


@dataclass
class MarkovRegressionModel:
    project_path: str
    project_data: ProjectData = None

    df: pd.DataFrame = None

    aic_list: list = None
    resid_list: list = None
    neuron_list: list = None
    results_list: list = None

    def __post_init__(self):

        project_data = ProjectData.load_final_project_data_from_config(self.project_path)
        self.project_data = project_data

        kwargs = dict(channel_mode='dr_over_r_20', min_nonnan=0.9, filter_mode='rolling_mean')
        self.df = project_data.calc_default_traces(interpolate_nan=True, **kwargs)

    @cached_property
    def speed(self):
        speed_with_outliers = self.project_data.worm_posture_class.worm_speed_fluorescence_fps_signed
        speed = pd.Series(speed_with_outliers)
        return speed

    @cached_property
    def vis_cfg(self):
        return self.project_data.project_config.get_visualization_config()

    def plot_no_neuron_markov_model(self, to_save=True):
        mod_speed = sm.tsa.MarkovRegression(self.speed, k_regimes=2)
        res_speed = mod_speed.fit()

        speed_pred = res_speed.predict()
        plt.figure(dpi=100)

        plt.plot(self.speed, label='speed')
        plt.plot(speed_pred, label='predicted speed')
        plt.legend()
        self.project_data.shade_axis_using_behavior()

        plt.ylabel("Speed (mm/s)")
        plt.xlabel("Time (Frames)")

        r, p = stats.pearsonr(self.speed, speed_pred)
        plt.title(f"Correlation: {r:.2f}")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'no_neurons.png', prepend_subfolder=True)
            plt.savefig(fname)

        plt.show()

    def calc_aic_feature_selected_neurons(self, num_iters=5):

        # Get features
        aic_list = []
        resid_list = []
        neuron_list = []

        remaining_neurons = get_names_from_df(self.df)
        trace_len = self.df.shape[0]
        previous_traces = []

        for i in tqdm(range(num_iters)):
            best_aic = 0
            best_resid = np.inf
            best_neuron = None

            for n in tqdm(remaining_neurons, leave=False):
                exog = pd.concat(previous_traces + [self.df[n]], axis=1)

                mod = sm.tsa.MarkovRegression(self.speed[:trace_len], k_regimes=2, exog=exog)
                res = mod.fit()

                if np.sum(res.resid ** 2) < best_resid:
                    best_resid = np.sum(res.resid ** 2)
                    best_aic = res.aic
                    best_neuron = n

            print(f"{best_neuron} selected for iteration {i}")
            aic_list.append(best_aic)
            resid_list.append(best_resid)
            neuron_list.append(best_neuron)

            previous_traces.append(self.df[best_neuron])
            remaining_neurons.remove(best_neuron)

        # Fit models
        results_list = []
        previous_traces = []
        for n in tqdm(neuron_list):
            exog = pd.concat(previous_traces + [self.df[n]], axis=1)
            mod = sm.tsa.MarkovRegression(self.speed[:trace_len], k_regimes=2, exog=exog)
            res = mod.fit()
            previous_traces.append(self.df[n])
            results_list.append(res)

        self.aic_list = aic_list
        self.resid_list = resid_list
        self.neuron_list = neuron_list
        self.results_list = results_list

    def plot_aic_feature_selected_neurons(self, to_save=True):
        if self.aic_list is None:
            self.calc_aic_feature_selected_neurons()

        aic_list = self.aic_list
        resid_list = self.resid_list
        neuron_list = self.neuron_list
        results_list = self.results_list
        trace_len = self.df.shape[0]

        # Plot 1
        fig, ax = plt.subplots(dpi=100)
        ax.plot(resid_list, label="Residual")

        ax2 = ax.twinx()
        ax2.plot(aic_list, label="AIC", c='tab:orange')

        ax.set_xticks(ticks=range(len(neuron_list)), labels=neuron_list, rotation=45)
        plt.xlabel("Neuron selected each iteration")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'error_across_neurons.png', prepend_subfolder=True)
            plt.savefig(fname)

        # Plot 2
        all_pred = [r.predict() for r in results_list]
        plt.figure(dpi=100)
        plt.plot(self.speed, label='speed', lw=2)

        for p, lab in zip(all_pred, neuron_list[0:8]):
            line = plt.plot(p, label=lab)
        plt.legend()
        plt.title("Predictions with cumulatively included neurons")
        plt.ylabel("Speed (mm/s)")
        plt.xlabel("Time (Frames)")

        r, p = stats.pearsonr(self.speed[:trace_len], all_pred[-1])
        plt.title(f"Best correlation: {r:.2f}")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'with_all_neurons_aic_feature_selected.png', prepend_subfolder=True)
            plt.savefig(fname)

        # Plot 3
        all_traces = [self.df[n] for n in neuron_list]

        plt.figure(dpi=100)
        for i, (t, lab) in enumerate(zip(all_traces, neuron_list[:5])):
            line = plt.plot(t - i, label=lab)
        plt.legend()

        self.project_data.shade_axis_using_behavior()
        plt.title("Neurons selected as predictive (top is best)")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'aic_predictive_traces.png', prepend_subfolder=True)
            plt.savefig(fname)

        plt.show()
