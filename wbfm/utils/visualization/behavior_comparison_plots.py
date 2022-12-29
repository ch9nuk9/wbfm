import logging
import os
import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Dict, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.exceptions
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import cross_validate, RepeatedKFold, train_test_split
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from tqdm.auto import tqdm
from wbfm.utils.general.utils_matplotlib import paired_boxplot_from_dataframes, corrfunc
from wbfm.utils.projects.finished_project_data import ProjectData
import statsmodels.api as sm
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project


@dataclass
class NeuronEncodingBase:
    """General class for behavioral encoding or correlations"""
    project_path: str

    dataframes_to_load: List[str] = field(default_factory=lambda: ['red', 'green', 'ratio', 'ratio_filt'])

    is_valid: bool = True
    df_kwargs: dict = field(default_factory=dict)

    @cached_property
    def project_data(self) -> ProjectData:
        return ProjectData.load_final_project_data_from_config(self.project_path)

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
                opt['filter_mode'] = 'bilateral'
            opt['channel_mode'] = channel_key
            all_dfs[key] = self.project_data.calc_default_traces(**opt, **self.df_kwargs)

        # Align columns to commmon subset
        all_column_names = [df.columns for df in all_dfs.values()]
        common_column_names = reduce(np.intersect1d, all_column_names)
        all_to_drop = [set(df.columns) - set(common_column_names) for df in all_dfs.values()]
        for key, to_drop in zip(all_dfs.keys(), all_to_drop):
            all_dfs[key].drop(columns=to_drop, inplace=True)

        print("Finished calculating traces!")

        return all_dfs


@dataclass
class NeuronToUnivariateEncoding(NeuronEncodingBase):
    """Subclass for specifically encoding a 1-d behavioral variable. By default this is speed"""

    cv: int = None

    _last_model_calculated: callable = None

    def __post_init__(self):
        self.df_kwargs['interpolate_nan'] = True

    def calc_multi_neuron_encoding(self, df_name, y_train=None):
        """Speed by default"""
        X = self.all_dfs[df_name]
        X_train, X_test, y_train, y_test, y_total, y_train_name = self._get_valid_test_train_split_from_name(X, y_train)
        alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)
            model = RidgeCV(cv=self.cv, alphas=alphas).fit(X_train, y_train)
        score = model.score(X_test, y_test)
        y_pred = model.predict(X)  # For entire dataset
        self._last_model_calculated = model
        return score, model, y_total, y_pred, y_train_name

    def calc_single_neuron_encoding(self, df_name, y_train=None):
        """
        Note that this does cross validation within the cross validation to select:
            ridge alpha (inner) and best neuron (outer)
        """
        X = self.all_dfs[df_name]
        X_train, X_test, y_train, y_test, y_total, y_train_name = self._get_valid_test_train_split_from_name(X, y_train)
        alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)
            estimator = RidgeCV(cv=self.cv, alphas=alphas)
            # This can be parallelized, but has a pickle error on my machine
            sfs = SequentialFeatureSelector(estimator=estimator,
                                            n_features_to_select=1, direction='forward', cv=self.cv)
            sfs.fit(X_train, y_train)
        # Refit the model on test data
        feature_names = get_names_from_df(self.all_dfs[df_name])
        best_neuron = feature_names[np.where(sfs.get_support())[0][0]]
        X_test_best_single_neuron = X_test[best_neuron].values.reshape(-1, 1)
        model = sfs.estimator.fit(X_test_best_single_neuron, y_test)
        score = model.score(X_test_best_single_neuron, y_test)

        X_best_single_neuron = X[best_neuron].values.reshape(-1, 1)
        y_pred = model.predict(X_best_single_neuron)  # Entire dataset, but still only one neuron
        self._last_model_calculated = model
        return score, model, y_total, y_pred, y_train_name, best_neuron

    def _get_valid_test_train_split_from_name(self, X, y_train_name) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        trace_len = X.shape[0]
        # Get 1d series from behavior
        if y_train_name is None:
            y_train_name = 'signed_speed'
            y = self.project_data.worm_posture_class.worm_speed_fluorescence_fps_signed[:trace_len]
        elif isinstance(y_train_name, str):
            if y_train_name == 'abs_speed':
                y = self.project_data.worm_posture_class.worm_speed_fluorescence_fps[:trace_len]
            elif y_train_name == 'leifer_curvature':
                y = self.project_data.worm_posture_class.leifer_curvature_from_kymograph[:trace_len]
            elif y_train_name == 'pirouette':
                y = self.project_data.worm_posture_class.calc_psuedo_pirouette_state()[:trace_len]
            else:
                raise NotImplementedError(y_train_name)
        else:
            raise NotImplementedError(y_train_name)
        y.reset_index(drop=True, inplace=True)

        # Remove nan points, if any
        valid_ind = np.where(~np.isnan(y))[0]
        X = X.iloc[valid_ind, :]
        y = y.iloc[valid_ind]
        
        # Build test train split
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        return X_train, X_test, y_train, y_test, y, y_train_name

    def plot_model_prediction(self, df_name, y_train=None, use_multineuron=True, **kwargs):
        """Plots model prediction over raw data"""
        if use_multineuron:
            score, model, y_total, y_pred, y_train_name = \
                self.calc_multi_neuron_encoding(df_name, y_train=y_train)
            y_name = f"multineuron_{y_train_name}"
        else:
            score, model, y_total, y_pred, y_train_name, _ = \
                self.calc_single_neuron_encoding(df_name, y_train=y_train)
            y_name = f"single_best_neuron_{y_train_name}"
        self._plot(df_name, y_pred, y_total, y_name=y_name, score=score, **kwargs)

    def plot_sorted_correlations(self, df_name, y_train=None, to_save=False, saving_folder=None):
        """
        Does not fit a model, just raw correlation
        """
        X = self.all_dfs[df_name]
        # Note: just use this function to resolve the name; do not actually use the train-test split
        _, _, _, _, y_total, y_train_name = self._get_valid_test_train_split_from_name(X, y_train)

        corr = X.corrwith(y_total)
        idx = np.argsort(corr)
        names = get_names_from_df(X)

        fig, ax = plt.subplots(dpi=200)
        x = range(len(idx))
        plt.bar(x, corr.iloc[idx.values])

        labels = np.array(names)[idx.values]
        labels = [name2int_neuron_and_tracklet(n) for n in labels]
        # plt.xticks(x, labels="")
        # ymin = np.min(corr) - 0.1
        # for i, name in enumerate(labels):
        #     plt.annotate(text=name, xy=(i, ymin), xytext=(i, ymin-0.1*(-i % 8)-0.1), xycoords='data', arrowprops={'width':1, 'headwidth':0}, annotation_clip=False)
        # ax.xaxis.set_major_locator(MultipleLocator(10))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        plt.xticks(ticks=x, labels=labels, fontsize=6)
        # ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
        plt.grid(which='major', axis='x')
        ax.set_axisbelow(True)
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            tick.set_pad(8 * (i % 4))
        plt.title(f"Sorted correlation: {df_name} traces with {y_train_name}")

        if to_save:
            fname = f"sorted_correlation_{df_name}_{y_train_name}.png"
            self._savefig(fname, saving_folder)

    def calc_dataset_summary_df(self, name: str, **kwargs) -> pd.DataFrame:
        """
        Calculates a summary number for the full dataset:
            The linear model error for a) the best single neuron and b) the multivariate encoding

        Parameters
        ----------
        name

        Returns
        -------

        """

        multi = self.calc_multi_neuron_encoding(name, **kwargs)[0]
        single = self.calc_single_neuron_encoding(name, **kwargs)[0]

        df_dict = {'best_single_neuron': single, 'multi_neuron': multi,
                   'dataset_name': self.project_data.shortened_name}
        df = pd.DataFrame(df_dict, index=[0])
        return df

    def plot_encoding_and_weights(self, df_name, y_train=None, y_name="speed"):
        """
        Plots the fit, regression weights, and grid plot for a Lasso model

        Parameters
        ----------
        df_name
        y_train
        y_name

        Returns
        -------

        """
        X_train = self.all_dfs[df_name]
        if y_train is None:
            y_train = self.project_data.worm_posture_class.worm_speed_fluorescence_fps_signed[:X_train.shape[0]]
        cv_model = self._plot_linear_regression_coefficients(X_train, y_train, df_name, y_name=y_name)[0]
        model = cv_model['estimator'][0]  # Just predict using a single model?
        y_pred = model.predict(X_train)
        self._plot(df_name, y_pred, y_train)

    def _plot(self, df_name, y_pred, y_train, y_name="", score=None, to_save=False, saving_folder=None):
        """
        Plots predictions and training data

        Assumes both y_pred and y_train are the length of the entire dataset (not a train-test split)

        Parameters
        ----------
        df_name
        y_pred
        y_train
        y_name
        score
        to_save
        saving_folder

        Returns
        -------

        """
        if score is None:
            score = median_absolute_error(y_train, y_pred)
        fig, ax = plt.subplots(dpi=200)
        opt = dict()
        if df_name == 'green' or df_name == 'red':
            opt['color'] = df_name
        ax.plot(y_pred, label='prediction', **opt)

        ax.set_title(f"Prediction error {score:.3f} from {df_name} traces ({self.project_data.shortened_name})")
        plt.ylabel(f"{y_name}")
        plt.xlabel("Time (volumes)")
        ax.plot(y_train, color='black', label='Target', alpha=0.8)
        plt.legend()
        self.project_data.shade_axis_using_behavior()

        if to_save:
            fname = f"regression_fit_{df_name}_{y_name}.png"
            self._savefig(fname, saving_folder)

    def _savefig(self, fname, saving_folder):
        if saving_folder is None:
            vis_cfg = self.project_data.project_config.get_visualization_config()
            fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        else:
            fname = os.path.join(saving_folder, f"{self.project_data.shortened_name}-{fname}")
        plt.savefig(fname)

    def _plot_linear_regression_coefficients(self, X, y, df_name, model=None,
                                             only_plot_nonzero=True, also_plot_traces=True, y_name="speed"):
        # From https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py
        if model is None:
            alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation
            model = LassoCV(alphas=alphas, max_iter=1000)

        feature_names = get_names_from_df(self.all_dfs[df_name])
        initial_val = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
        cv_model = cross_validate(
            model,
            X,
            y,
            cv=cv,
            return_estimator=True,
            n_jobs=2,
        )
        coefs = pd.DataFrame(
            [est.coef_ for est in cv_model["estimator"]], columns=feature_names
        )

        # Only keep neurons with nonzero values
        tol = 1e-3
        if only_plot_nonzero:
            coefs = coefs.loc[:, coefs.mean().abs() > tol]

        # Boxplot of variability
        plt.figure(dpi=100)
        sns.stripplot(data=coefs, orient="h", color="k", alpha=0.5, linewidth=1)
        sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=100)
        plt.axvline(x=0, color=".5")
        title_str = f"Coefficient variability for {self.project_data.shortened_name}"
        plt.title(title_str)
        plt.subplots_adjust(left=0.3)
        plt.grid(axis='y', which='both')

        vis_cfg = self.project_data.project_config.get_visualization_config()
        fname = vis_cfg.resolve_relative_path(f"lasso_coefficients_{df_name}_{y_name}.png", prepend_subfolder=True)
        plt.savefig(fname)

        # gridplot of traces
        if also_plot_traces:
            direct_shading_dict = coefs.mean().to_dict()
            make_grid_plot_using_project(self.project_data, 'ratio', 'integration',
                                         neuron_names_to_plot=get_names_from_df(coefs),
                                         direct_shading_dict=direct_shading_dict,
                                         sort_using_shade_value=True, savename_suffix=f"{y_name}_encoding")

        os.environ["PYTHONWARNINGS"] = initial_val  # Also affect subprocesses

        return cv_model, coefs


@dataclass
class NeuronToMultivariateEncoding(NeuronEncodingBase):
    """Designed for single-neuron correlations to all kymograph body segments"""

    def __post_init__(self):

        if self.project_data.worm_posture_class.has_full_kymograph and self.project_data.has_traces():
            self.is_valid = True
        else:
            logging.warning("Kymograph not found, this class will not work")
            self.is_valid = False

    @cached_property
    def all_dfs_corr(self) -> Dict[str, pd.DataFrame]:
        kymo = self.project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)
        kymo_smaller = kymo.loc[:, 3:60].copy()

        all_dfs_corr = {key: pd.concat([df, kymo_smaller], axis=1).corr() for key, df in self.all_dfs.items()}

        # Only get the corner we care about: traces vs. kymo
        all_dfs_corr = self.get_corner_from_corr_df(all_dfs_corr)
        return all_dfs_corr

    @cached_property
    def all_dfs_corr_fwd(self) -> Dict[str, pd.DataFrame]:
        assert self.project_data.worm_posture_class.has_beh_annotation, "Behavior annotations required"

        kymo = self.project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)

        # New: only do certain indices
        ind = self.project_data.worm_posture_class.beh_annotation == 0
        kymo_smaller = kymo.loc[ind, 3:60].copy()
        all_dfs_corr = {key: pd.concat([df.loc[ind, :], kymo_smaller], axis=1).corr() for key, df in self.all_dfs.items()}

        # Only get the corner we care about: traces vs. kymo
        all_dfs_corr = self.get_corner_from_corr_df(all_dfs_corr)
        return all_dfs_corr

    @cached_property
    def all_dfs_corr_rev(self) -> Dict[str, pd.DataFrame]:
        assert self.project_data.worm_posture_class.has_beh_annotation, "Behavior annotations required"

        kymo = self.project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)

        # New: only do certain indices
        ind = self.project_data.worm_posture_class.beh_annotation == 1
        kymo_smaller = kymo.loc[ind, 3:60].copy()
        all_dfs_corr = {key: pd.concat([df.loc[ind, :], kymo_smaller], axis=1).corr() for key, df in self.all_dfs.items()}

        # Only get the corner we care about: traces vs. kymo
        all_dfs_corr = self.get_corner_from_corr_df(all_dfs_corr)
        return all_dfs_corr

    def get_corner_from_corr_df(self, all_dfs_corr):
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

    def calc_per_neuron_df(self, name: str) -> pd.DataFrame:
        """
        Calculates a summary dataframe of information per neuron.
            Rows: neuron names
            Columns: ['median_brightness', 'var_brightness', 'body_segment_argmax', 'corr_max', 'dataset_name']

            Note that dataset_name is used when this is concatenated with other dataframes

        Parameters
        ----------
        name - str, one of self.all_labels

        Returns
        -------

        """
        df_corr = self.all_dfs_corr[name]
        df_traces = self.all_dfs[name]

        body_segment_argmax = df_corr.columns[df_corr.abs().apply(pd.Series.argmax, axis=1)]
        body_segment_argmax = pd.Series(body_segment_argmax, index=df_corr.index)

        corr_max = df_corr.abs().max(axis=1)
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
        df_start = self.calc_per_neuron_df(start_name)
        df_final = self.calc_per_neuron_df(final_name)
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
        df_start = self.calc_per_neuron_df(start_name)
        df_final = self.calc_per_neuron_df(final_name)

        # Build columns and join
        df_start['source_data'] = start_name
        df_final['source_data'] = final_name

        df = pd.concat([df_start, df_final], axis=0)
        df.source_data = df.source_data.astype('category')
        df = df.reset_index().rename(columns={'index': 'neuron_name'})

        return df

    def plot_correlation_of_examples(self, to_save=True, only_within_state=None, **kwargs):
        # Calculate correlation dataframes
        if only_within_state is None:
            all_dfs = list(self.all_dfs_corr.values())
        elif only_within_state.lower() == 'fwd':
            all_dfs = list(self.all_dfs_corr_fwd.values())
        elif only_within_state.lower() == 'rev':
            all_dfs = list(self.all_dfs_corr_rev.values())
        else:
            raise NotImplementedError

        self._multi_plot(list(self.all_dfs.values()), all_dfs,
                         self.all_labels, self.all_colors,
                         project_data=self.project_data, to_save=to_save, **kwargs)

    def plot_correlation_of_prefentially_one_state(self, to_save=True, only_within_state=None, **kwargs):
        # Calculate correlation dataframes
        all_dfs_fwd = list(self.all_dfs_corr_fwd.values())
        all_dfs_rev = list(self.all_dfs_corr_rev.values())
        if only_within_state.lower() == 'fwd':
            all_dfs = [f - r for f, r in zip(all_dfs_fwd, all_dfs_rev)]
        elif only_within_state.lower() == 'rev':
            all_dfs = [r - f for f, r in zip(all_dfs_fwd, all_dfs_rev)]
        else:
            raise NotImplementedError

        all_figs = self._multi_plot(list(self.all_dfs.values()), all_dfs,
                                    self.all_labels, self.all_colors,
                                    project_data=self.project_data, to_save=to_save, **kwargs)
        # for fig in all_figs:
        #     fig.axes[0][0].set_ylabel("Differential correlation")

    def plot_correlation_histograms(self, to_save=True):
        plt.figure(dpi=100)
        all_max_corrs = [df_corr.abs().max(axis=1) for df_corr in self.all_dfs_corr.values()]

        plt.hist(all_max_corrs,
                 color=self.all_colors,
                 label=self.all_labels)
        plt.xlim(-0.2, 1)
        plt.title(self.project_data.shortened_name)
        plt.xlabel("Maximum correlation")
        plt.legend()

        if to_save:
            vis_cfg = self.project_data.project_config.get_visualization_config()
            fname = f'maximum_correlation_kymograph_histogram.png'
            fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

            plt.savefig(fname)

    def plot_histogram_difference_after_ratio(self, df_start_names=None, df_final_name='ratio', to_save=True):
        plt.figure(dpi=100)

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
        title_str = f"Correlation difference between {df_start_names} to {df_final_name}"
        plt.title(title_str)

        if to_save:
            vis_cfg = self.project_data.project_config.get_visualization_config()
            fname = f'{title_str}.png'
            fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            plt.savefig(fname)

    def plot_paired_boxplot_difference_after_ratio(self, df_start_name='red', df_final_name='ratio', to_save=True):
        plt.figure(dpi=100)
        # Get data
        both_maxes = self.get_data_for_paired_boxplot(df_final_name, df_start_name)

        # Plot
        paired_boxplot_from_dataframes(both_maxes, [df_start_name, df_final_name])

        plt.ylim(0, 0.8)
        plt.ylabel("Absolute correlation")
        title_str = f"Change in correlation from {df_start_name} to {df_final_name}"
        plt.title(title_str)

        if to_save:
            vis_cfg = self.project_data.project_config.get_visualization_config()
            fname = f'{title_str}.png'
            fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            plt.savefig(fname)

    def get_data_for_paired_boxplot(self, df_final_name, df_start_name):
        df_start = self.all_dfs_corr[df_start_name]
        df_final = self.all_dfs_corr[df_final_name]
        start_maxes = df_start.max(axis=1)
        final_max = df_final.max(axis=1)
        both_maxes = pd.concat([start_maxes, final_max], axis=1).T
        return both_maxes

    def plot_phase_difference(self, df_start_name='red', df_final_name='green', corr_thresh=0.2, remove_zeros=True,
                              to_save=True):
        """
        Green minus red

        Returns
        -------

        """
        plt.figure(dpi=100)
        df_start = self.all_dfs_corr[df_start_name].copy()
        df_final = self.all_dfs_corr[df_final_name].copy()

        ind_to_keep = df_start.abs().max(axis=1) > corr_thresh
        df_start = df_start.loc[ind_to_keep, :]
        df_final = df_final.loc[ind_to_keep, :]

        start_body_segment_argmax = df_start.columns[df_start.abs().apply(pd.Series.argmax, axis=1)]
        final_body_segment_argmax = df_final.columns[df_final.abs().apply(pd.Series.argmax, axis=1)]

        diff = final_body_segment_argmax - start_body_segment_argmax
        title_str = f"{df_final_name} - {df_start_name} with starting corr > {corr_thresh}"
        if remove_zeros:
            diff = diff[diff != 0]
            title_str = f"{title_str} (zeros removed)"
        plt.hist(diff, bins=np.arange(diff.min(), diff.max()))
        plt.title(title_str)
        plt.xlabel("Phase shift (body segments)")

        if to_save:
            vis_cfg = self.project_data.project_config.get_visualization_config()
            fname = f'{title_str.replace(">", "ge")}.png'
            fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            plt.savefig(fname)

    @staticmethod
    def _multi_plot(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations=None,
                    project_data: ProjectData=None,
                    corr_thresh=0.3, which_df_to_apply_corr_thresh=-1, max_num_plots=None,
                    xlim=None, to_save=False, all_names=None):
        all_figs = []
        if xlim is None:
            xlim = [100, 450]
        if ax_locations is None:
            ax_locations = [1, 1, 3, 3, 3]

        if all_names is None:
            all_names = list(all_dfs_corr_list[0].index)
        else:
            # Plot all that are sent
            corr_thresh = None
        num_open_plots = 0

        for i in range(all_names):
            abs_corr = all_dfs_corr_list[which_df_to_apply_corr_thresh].iloc[i, :]
            if corr_thresh is not None and abs_corr.max() < corr_thresh:
                continue
            else:
                num_open_plots += 1

            fig, axes = plt.subplots(ncols=2, nrows=2, dpi=100, figsize=(15, 5))
            all_figs.append(fig)
            axes = np.ravel(axes)
            neuron_name = all_names[i]

            for df, df_corr, lab, col, ax_loc in zip(all_dfs_list, all_dfs_corr_list, all_labels, all_colors, ax_locations):

                plt_opt = dict(label=lab, color=col)
                # Always put the correlation on ax 0
                abs_corr = df_corr.iloc[i, :]
                axes[0].plot(abs_corr, **plt_opt)

                # Put the trace on the passed axis
                trace = df[neuron_name]
                axes[ax_loc].plot(trace / trace.mean(), **plt_opt)

            axes[0].set_xlabel("Body segment")
            axes[0].set_ylabel("Correlation")
            axes[0].set_ylim(-0.75, 0.75)
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
        return all_figs


@dataclass
class MultiProjectBehaviorPlotter:
    all_project_paths: list

    class_constructor: callable = NeuronToMultivariateEncoding
    use_threading: bool = True

    _all_behavior_plotters: List[NeuronEncodingBase] = None

    def __post_init__(self):
        # Initialize the behavior plotters
        self._all_behavior_plotters = [self.class_constructor(p) for p in self.all_project_paths]

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
        # plt.ylim(0, 0.8)

    def __repr__(self):
        return f"Multiproject analyzer with {len(self._all_behavior_plotters)} projects"


@dataclass
class MarkovRegressionModel:
    project_path: str

    behavior_to_predict: str = 'speed'

    project_data: ProjectData = None
    df: pd.DataFrame = None

    aic_list: list = None
    resid_list: list = None
    neuron_list: list = None
    results_list: list = None

    def __post_init__(self):

        project_data = ProjectData.load_final_project_data_from_config(self.project_path)
        self.project_data = project_data

        kwargs = dict(channel_mode='dr_over_r_20', min_nonnan=0.9, filter_mode='bilateral')
        self.df = project_data.calc_default_traces(interpolate_nan=True, **kwargs)

    def get_valid_ind_and_trace(self) -> Tuple[np.ndarray, pd.Series]:
        if self.behavior_to_predict == 'speed':
            trace = self.project_data.worm_posture_class.worm_speed_fluorescence_fps_signed
            trace = pd.Series(trace)
        elif self.behavior_to_predict == 'leifer_curvature':
            worm = self.project_data.worm_posture_class
            trace = worm.leifer_curvature_from_kymograph[worm.subsample_indices].copy().reset_index(drop=True)
        else:
            raise NotImplementedError(self.behavior_to_predict)

        valid_ind = np.where(~np.isnan(trace))[0]
        valid_ind = valid_ind[valid_ind < self.trace_len]

        return valid_ind, trace[valid_ind]

    @property
    def trace_len(self):
        return self.df.shape[0]

    @cached_property
    def vis_cfg(self):
        return self.project_data.project_config.get_visualization_config()

    def plot_no_neuron_markov_model(self, to_save=True):
        valid_ind, trace = self.get_valid_ind_and_trace()
        mod = sm.tsa.MarkovRegression(trace, k_regimes=2)
        res = mod.fit()
        pred = res.predict()

        plt.figure(dpi=100)
        plt.plot(trace, label=self.behavior_to_predict)
        plt.plot(pred, label=f'predicted {self.behavior_to_predict}')
        plt.legend()
        self.project_data.shade_axis_using_behavior()

        plt.ylabel(f"{self.behavior_to_predict}")
        plt.xlabel("Time (Frames)")
        r = trace.corr(pred)
        plt.title(f"Correlation: {r:.2f}")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'{self.behavior_to_predict}_no_neurons.png', prepend_subfolder=True)
            plt.savefig(fname)

        plt.show()

    def calc_aic_feature_selected_neurons(self, num_iters=4):
        valid_ind, trace = self.get_valid_ind_and_trace()
        # Get features
        aic_list = []
        resid_list = []
        neuron_list = []

        remaining_neurons = get_names_from_df(self.df)
        previous_traces = []

        for i in tqdm(range(num_iters)):
            best_aic = 0
            best_resid = np.inf
            best_neuron = None

            for n in tqdm(remaining_neurons, leave=False):
                exog = pd.concat(previous_traces + [self.df[n][valid_ind]], axis=1)

                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=ConvergenceWarning)
                    warnings.simplefilter(action='ignore', category=ValueWarning)
                    warnings.simplefilter(action='ignore', category=RuntimeWarning)
                    mod = sm.tsa.MarkovRegression(trace, k_regimes=2, exog=exog)
                    res = mod.fit()

                if np.sum(res.resid ** 2) < best_resid:
                    best_resid = np.sum(res.resid ** 2)
                    best_aic = res.aic
                    best_neuron = n

            print(f"{best_neuron} selected for iteration {i}")
            aic_list.append(best_aic)
            resid_list.append(best_resid)
            neuron_list.append(best_neuron)
            previous_traces.append(self.df[best_neuron][valid_ind])
            remaining_neurons.remove(best_neuron)

        # Fit models
        results_list = []
        previous_traces = []
        for n in tqdm(neuron_list):
            exog = pd.concat(previous_traces + [self.df[n][valid_ind]], axis=1)
            mod = sm.tsa.MarkovRegression(trace, k_regimes=2, exog=exog)
            res = mod.fit()
            previous_traces.append(self.df[n][valid_ind])
            results_list.append(res)

        self.aic_list = aic_list
        self.resid_list = resid_list
        self.neuron_list = neuron_list
        self.results_list = results_list

    def plot_aic_feature_selected_neurons(self, to_save=True):
        valid_ind, trace = self.get_valid_ind_and_trace()
        if self.aic_list is None:
            self.calc_aic_feature_selected_neurons()
        aic_list = self.aic_list
        resid_list = self.resid_list
        neuron_list = self.neuron_list
        results_list = self.results_list

        # Plot 1
        fig, ax = plt.subplots(dpi=100)
        ax.plot(resid_list, label="Residual")

        ax2 = ax.twinx()
        ax2.plot(aic_list, label="AIC", c='tab:orange')

        ax.set_xticks(ticks=range(len(neuron_list)), labels=neuron_list, rotation=45)
        plt.xlabel("Neuron selected each iteration")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(f'{self.behavior_to_predict}_error_across_neurons.png',
                                                       prepend_subfolder=True)
            plt.savefig(fname)

        # Plot 2
        all_pred = [r.predict() for r in results_list]
        plt.figure(dpi=100)
        plt.plot(trace, label=self.behavior_to_predict, lw=2)

        for p, lab in zip(all_pred, neuron_list[0:8]):
            line = plt.plot(p, label=lab)
        plt.legend()
        plt.title("Predictions with cumulatively included neurons")
        plt.ylabel(f"{self.behavior_to_predict}")
        plt.xlabel("Time (Frames)")
        self.project_data.shade_axis_using_behavior()

        r = trace.corr(all_pred[-1])
        plt.title(f"Best correlation: {r:.2f}")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(
                f'{self.behavior_to_predict}_with_all_neurons_aic_feature_selected.png', prepend_subfolder=True)
            plt.savefig(fname)

        # Plot 3
        all_traces = [self.df[n][valid_ind] for n in neuron_list]

        plt.figure(dpi=100)
        for i, (t, lab) in enumerate(zip(all_traces, neuron_list[:5])):
            line = plt.plot(t - i, label=lab)
        plt.legend()
        self.project_data.shade_axis_using_behavior()
        plt.title("Neurons selected as predictive (top is best)")

        if to_save:
            fname = self.vis_cfg.resolve_relative_path(
                f'{self.behavior_to_predict}_aic_predictive_traces.png', prepend_subfolder=True)
            plt.savefig(fname)

        plt.show()
