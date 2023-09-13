import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ipywidgets import interact
from sklearn.cross_decomposition import CCA
import plotly.express as px
from sklearn.decomposition import PCA

from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
import plotly.graph_objects as go
from methodtools import lru_cache
import cca_zoo.models as scc_mod

from wbfm.utils.projects.finished_project_data import ProjectData


@dataclass
class CCAPlotter:
    """
    Dataclass for CCA visualization
    """

    project_data: ProjectData

    _df_traces: pd.DataFrame = None
    _df_beh: pd.DataFrame = None
    df_beh_binary: pd.DataFrame = None

    # Preprocessing options. Goal: more interpretable CCA weights
    preprocess_traces_using_pca: bool = True
    truncate_traces_to_n_components: int = None  # Not used if preprocess_traces_using_pca is False
    preprocess_behavior_using_pca: bool = True
    truncate_behavior_to_n_components: int = None  # Not used if preprocess_behavior_using_pca is False
    _df_traces_truncated: pd.DataFrame = None
    _df_beh_truncated: pd.DataFrame = None
    _pca_traces: PCA = None
    _pca_beh: PCA = None

    def __post_init__(self):
        # Default traces and behaviors
        opt = dict(filter_mode='rolling_mean', interpolate_nan=True, nan_tracking_failure_points=True,
                   use_physical_time=True, rename_neurons_using_manual_ids=True)

        df_traces = self.project_data.calc_default_traces(**opt)
        self._df_traces = df_traces
        if self.preprocess_traces_using_pca:
            X, pca = self._truncate_using_pca(df_traces, n_components=self.truncate_traces_to_n_components)
            self._df_traces_truncated = pd.DataFrame(X, index=df_traces.index)
            self._pca_traces = pca

        df_beh = self.project_data.calc_default_behaviors(**opt, add_constant=False)
        # Standardize, but do not fully z-score, the behaviors
        beh_std = df_beh.std()
        beh_std[beh_std == 0] = 1  # There may be a constant column, which should not be divided by 0
        df_beh = df_beh / beh_std
        self._df_beh = df_beh
        if self.preprocess_behavior_using_pca:
            X, pca = self._truncate_using_pca(df_beh, n_components=self.truncate_behavior_to_n_components)
            self._df_beh_truncated = pd.DataFrame(X, index=df_beh.index)
            self._pca_beh = pca
        # No filtering
        self.df_beh_binary = self.project_data.calc_default_behaviors(binary_behaviors=True)

    def _truncate_using_pca(self, df_traces, n_components=None):
        X = fill_nan_in_dataframe(df_traces, do_filtering=False)
        X -= X.mean()
        pca = PCA(n_components=n_components, whiten=False)
        X = pca.fit_transform(X)
        return X, pca

    @property
    def df_traces(self):
        if self.preprocess_traces_using_pca:
            return self._df_traces_truncated
        else:
            return self._df_traces

    @property
    def df_beh(self):
        if self.preprocess_behavior_using_pca:
            return self._df_beh_truncated
        else:
            return self._df_beh

    @lru_cache(maxsize=16)
    def calc_cca(self, n_components=3, binary_behaviors=False, sparse_tau=None):
        """
        Does regular or sparse CCA. Returns the transformed data and the CCA object

        if sparse_tau is None, then regular CCA is performed (default)
        Note that because this is a cached function, sparse_tau should be a tuple, not a list

        Parameters
        ----------
        n_components
        binary_behaviors
        sparse_tau

        Returns
        -------

        """

        X = self.df_traces
        Y = self._get_beh_df(binary_behaviors)

        if sparse_tau is None:
            cca = CCA(n_components=n_components)
            X_r, Y_r = cca.fit_transform(X, Y)
        else:
            cca = scc_mod.SCCA_IPLS(latent_dims=n_components, tau=sparse_tau)
            X_r, Y_r = cca.fit_transform([X, Y])

        return X_r, Y_r, cca

    def calc_cca_reconstruction(self, **kwargs):
        X_r, Y_r, cca = self.calc_cca(**kwargs)
        return cca.inverse_transform(X_r, Y_r)

    def calc_r_squared(self, use_pca=False, n_components=1, **kwargs):
        # First, calculate the reconstruction
        X = self._df_traces  # Use the non-truncated traces
        if use_pca:
            # See calc_pca_modes
            X = fill_nan_in_dataframe(X, do_filtering=False)
            X -= X.mean()
            pca = PCA(n_components=n_components, whiten=False)
            X_r_recon = pca.inverse_transform(pca.fit_transform(X))
        else:
            X_r_recon, _ = self.calc_cca_reconstruction(n_components=n_components, **kwargs)
            if self.preprocess_traces_using_pca:
                # Transform the reconstructed traces back to the original space
                X_r_recon = self._pca_traces.inverse_transform(X_r_recon)

        # Then, calculate the r-squared
        residual_variance = (X - X_r_recon).var().sum()
        total_variance = X.var().sum()

        return 1 - residual_variance / total_variance

    def _get_beh_df(self, binary_behaviors, raw_not_truncated=False):
        if binary_behaviors:
            Y = self.df_beh_binary
        else:
            if raw_not_truncated:
                Y = self._df_beh
            else:
                # This will be truncated if the behavior is preprocessed using PCA
                Y = self.df_beh
        return Y

    def visualize_modes(self, i=1, binary_behaviors=False, **kwargs):
        X_r, Y_r, cca = self.calc_cca(n_components=i, binary_behaviors=binary_behaviors, **kwargs)

        df = pd.DataFrame({'Latent X': X_r[:, i], 'Latent Y': Y_r[:, i]})
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.show()

        return fig

    def visualize_modes_and_weights(self, n_components=1, binary_behaviors=False,
                                    sort_trace_weights=True, sort_beh_weights=True,
                                    **kwargs):

        X_r, Y_r, cca = self.calc_cca(n_components=n_components, binary_behaviors=binary_behaviors, **kwargs)
        df_x, df_y = self.get_weights_from_cca(cca, binary_behaviors, **kwargs)

        def f(i=0):
            df = pd.DataFrame({'Latent X': X_r[:, i] / X_r[:, i].max(),
                               'Latent Y': Y_r[:, i] / Y_r[:, i].max()})
            fig = px.line(df)
            self.project_data.shade_axis_using_behavior(plotly_fig=fig)
            fig.show()

            y = df_y.iloc[i, :]
            if sort_beh_weights:
                y = y.sort_values(ascending=False)
            fig = px.bar(y)
            fig.show()

            x = df_x.iloc[i, :]
            if sort_trace_weights:
                x = x.sort_values(ascending=False)
            fig = px.bar(x)
            fig.show()

        interact(f, i=(0, X_r.shape[1] - 1))

    def get_weights_from_cca(self, cca, binary_behaviors, **kwargs):
        """Returns trace and behavioral weight dataframes from a CCA object, with the correct column names"""
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces
        if 'sparse_tau' in kwargs:
            df_y = pd.DataFrame(cca.weights[1], index=df_beh.columns).T
            df_x = pd.DataFrame(cca.weights[0], index=df_traces.columns).T
        else:
            df_y = pd.DataFrame(cca.y_weights_, index=df_beh.columns).T
            df_x = pd.DataFrame(cca.x_weights_, index=df_traces.columns).T
        # Convert the weights to the original neuron space, if using PCA preprocessing
        if self.preprocess_traces_using_pca:
            df_x = pd.DataFrame(self._pca_traces.inverse_transform(df_x), columns=self._df_traces.columns)
        if self.preprocess_behavior_using_pca:
            df_y = pd.DataFrame(self._pca_beh.inverse_transform(df_y), columns=self._df_beh.columns)
        return df_x, df_y

    def plot_single_mode(self, i_mode=0, binary_behaviors=False, use_pca=False, output_folder=None, **kwargs):

        if use_pca:
            X_r = self.project_data.calc_pca_modes(n_components=i_mode+1)
            df = pd.DataFrame({f'PCA mode {i_mode+1}': X_r[:, i_mode] / X_r[:, i_mode].max()})
        else:
            X_r, Y_r, cca = self.calc_cca(binary_behaviors=binary_behaviors, **kwargs)

            df = pd.DataFrame({f'Latent trace mode {i_mode+1}': X_r[:, i_mode] / X_r[:, i_mode].max(),
                               f'Latent behavior mode {i_mode+1}': Y_r[:, i_mode] / Y_r[:, i_mode].max()})
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.show()

        if output_folder is not None:
            fname = self._get_fig_filename(binary_behaviors, plot_3d=False, use_pca=use_pca, single_mode=True)
            fname = os.path.join(output_folder, fname)
            fig.write_html(fname)
            fname = fname.replace('.html', '.png')
            fig.write_image(fname)

    def plot(self, binary_behaviors=False, modes_to_plot=None, use_pca=False, use_X_r=True, sparse_tau=None,
             plot_3d=True, output_folder=None, DEBUG=False,
             ethogram_cmap_kwargs=None, beh_annotation_kwargs=None):
        if ethogram_cmap_kwargs is None:
            ethogram_cmap_kwargs = {}
        if beh_annotation_kwargs is None:
            beh_annotation_kwargs = {}
        if modes_to_plot is None:
            modes_to_plot = [0, 1, 2]
        if use_pca:
            X_r = self.project_data.calc_pca_modes(n_components=3)
            df_latents = pd.DataFrame(X_r)
        else:
            X_r, Y_r, cca = self.calc_cca(n_components=3, binary_behaviors=binary_behaviors, sparse_tau=sparse_tau)
            if use_X_r:
                df_latents = pd.DataFrame(X_r)
            else:
                df_latents = pd.DataFrame(Y_r)

        # Color the lines by behavior annotation (not necessarily the same as the behavior used for CCA)
        beh_annotation = dict(fluorescence_fps=True, reset_index=True, include_collision=False, include_turns=True,
                              include_head_cast=False, include_pause=False, include_slowing=False)
        beh_annotation.update(beh_annotation_kwargs)
        df_latents['state'] = self.project_data.worm_posture_class.beh_annotation(**beh_annotation)
        ethogram_cmap_kwargs.setdefault('include_turns', beh_annotation['include_turns'])
        ethogram_cmap_kwargs.setdefault('include_quiescence', beh_annotation['include_pause'])
        ethogram_cmap_kwargs.setdefault('include_collision', beh_annotation['include_collision'])
        ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs)
        df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_latents, modes_to_plot, 'state')
        state_codes = df_latents['state'].unique()

        if DEBUG:
            print(state_codes, ethogram_cmap, ethogram_cmap_kwargs)

        # Loop over behaviorally-colored short segments and plot
        phase_plot_list = []
        for i, state_code in enumerate(state_codes):
            if plot_3d:
                phase_plot_list.append(
                    go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                                 mode='lines', name=state_code.full_name,
                                 line=dict(color=ethogram_cmap.get(state_code, None), width=4)))
            else:
                phase_plot_list.append(
                    go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]],
                               mode='lines', name=state_code.full_name,
                               line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

        fig = go.Figure(layout=dict(height=1000, width=1000))
        fig.add_traces(phase_plot_list)

        # Hacky: https://community.plotly.com/t/scatter3d-background-plot-color/38838/4
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0, 0, 0, 0)",
                    tickvals=[-1, 0, 1],
                    showbackground=True,
                    gridcolor='black',
                    zerolinecolor="white",
                    title='Mode 1'
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0, 0, 0, 0)",
                    tickvals=[-1, 0, 1],
                    showbackground=True,
                    gridcolor='black',
                    zerolinecolor="white",
                    title='Mode 2'),
            )
        )
        if plot_3d:
            fig.update_layout(
                scene=dict(
                    zaxis=dict(
                        backgroundcolor="rgba(0, 0, 0,0)",
                        tickvals=[-1, 0, 1],
                        showbackground=True,
                        gridcolor='black',
                        zerolinecolor="white",
                        title='Mode 3'),
                ),
                # From: https://stackoverflow.com/questions/73187799/truncated-figure-with-plotly?noredirect=1#comment129258910_73187799
                # Note that this is hard to do in jupyter and then see the settings, but can be done with dash:
                # https://community.plotly.com/t/how-to-get-change-current-scene-camera-in-3d-plot-inside-jupyter-notebook-python/1912/4
                scene_camera=dict(eye=dict(x=1.0, y=1.0, z=2.5))
            )
        # Transparent background
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if output_folder is not None:
            fname = self._get_fig_filename(binary_behaviors, plot_3d, use_pca, single_mode=False)
            fname = os.path.join(output_folder, fname)

            fig.write_html(fname)
            fname = fname.replace('.html', '.png')
            fig.write_image(fname)

        fig.show()
        return fig

    def _get_fig_filename(self, binary_behaviors, plot_3d, use_pca, single_mode):
        # Build name based on options used
        if use_pca:
            fname = 'pca'
        else:
            fname = 'cca'
            if binary_behaviors:
                fname += '-binary'
            else:
                fname += '-continuous'
        if plot_3d:
            fname += '-3d'
        else:
            fname += '-2d'
        if single_mode:
            fname += '-mode'
        fname += '.html'
        return fname

    def calc_mode_correlation(self, **kwargs):
        """
        Calculates the correlation between the modes of the traces and behaviors, which is what CCA is maximizing

        Returns
        -------

        """

        X_r, Y_r, cca = self.calc_cca(**kwargs)
        corr_list = []
        for i in range(X_r.shape[1]):
            corr_list.append(pd.Series(X_r[:, i]).corr(pd.Series(Y_r[:, i])))
        return corr_list


def calc_r_squared_for_all_projects(all_projects, r_squared_kwargs=None, **kwargs):
    if r_squared_kwargs is None:
        r_squared_kwargs = {}
    all_cca_classes = {}
    all_r_squared = defaultdict(dict)

    opt_dict = {'pca': dict(use_pca=True),
                'cca': dict(use_pca=False),
                'cca_binary': dict(use_pca=False, binary_behaviors=True)}

    for name, p in all_projects.items():
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(r_squared_kwargs)
            all_r_squared[name][opt_name] = cca_plotter.calc_r_squared(**opt)

    df_r_squared = pd.DataFrame(all_r_squared).T

    return all_cca_classes, df_r_squared


def calc_mode_correlation_for_all_projects(all_projects, correlation_kwargs=None, **kwargs):
    if correlation_kwargs is None:
        correlation_kwargs = {}
    all_cca_classes = {}
    all_mode_correlations = defaultdict(dict)

    opt_dict = {'cca': dict(),
                'cca_binary': dict(binary_behaviors=True)}

    for name, p in all_projects.items():
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(correlation_kwargs)
            all_mode_correlations[opt_name][name] = cca_plotter.calc_mode_correlation(**opt)

    df_mode_correlations = pd.DataFrame(all_mode_correlations['cca'])
    df_mode_correlations_binary = pd.DataFrame(all_mode_correlations['cca_binary'])

    return all_cca_classes, df_mode_correlations, df_mode_correlations_binary


def calc_cca_weights_for_all_projects(all_projects, which_mode=0, weights_kwargs=None,
                                      neural_not_behavioral=True, correct_sign_using_top_weight=True,
                                      drop_unlabeled_neurons=True, min_datasets_present=5,
                                      **kwargs):
    if weights_kwargs is None:
        weights_kwargs = {}
    all_cca_classes = {}
    all_weights = defaultdict(dict)

    opt_dict = {'cca': dict(binary_behaviors=False),
                'cca_binary': dict(binary_behaviors=True)}

    for name, p in all_projects.items():
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(weights_kwargs)
            _, _, cca = cca_plotter.calc_cca(**opt)
            trace_weights, behavior_weights = cca_plotter.get_weights_from_cca(cca, **opt)
            if neural_not_behavioral:
                all_weights[opt_name][name] = trace_weights.loc[which_mode, :]
            else:
                all_weights[opt_name][name] = behavior_weights.loc[which_mode, :]

    df_weights = pd.DataFrame(all_weights['cca']).T
    df_weights_binary = pd.DataFrame(all_weights['cca_binary']).T

    if neural_not_behavioral:
        # Drop all neurons that contain 'neuron' in the name
        if drop_unlabeled_neurons:
            df_weights = df_weights.loc[:, ~df_weights.columns.str.contains('neuron')]
            df_weights_binary = df_weights_binary.loc[:, ~df_weights_binary.columns.str.contains('neuron')]
        # Remove neurons that are not present in at least min_datasets_present datasets
        if min_datasets_present > 0:
            df_weights = df_weights.loc[:, df_weights.notnull().sum() >= min_datasets_present]
            df_weights_binary = df_weights_binary.loc[:, df_weights_binary.notnull().sum() >= min_datasets_present]

    # The weights have sign ambiguity. Correct this by setting the top weight to be positive
    if correct_sign_using_top_weight:
        sign_vec = np.sign(df_weights.iloc[:, df_weights.abs().sum().argmax()])
        df_weights = df_weights.mul(sign_vec, axis='index')
        sign_vec = np.sign(df_weights.iloc[:, df_weights_binary.abs().sum().argmax()])
        df_weights_binary = df_weights_binary.mul(np.sign(df_weights_binary.iloc[:, 0]), axis='index')

    # Sort them by the signed median weight
    df_weights = df_weights.reindex(df_weights.median().sort_values(ascending=False).index, axis=1)
    df_weights_binary = df_weights_binary.reindex(df_weights_binary.median().sort_values(ascending=False).index, axis=1)

    return all_cca_classes, df_weights, df_weights_binary
