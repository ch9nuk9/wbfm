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

    df_traces: pd.DataFrame = None
    df_beh: pd.DataFrame = None
    df_beh_binary: pd.DataFrame = None

    def __post_init__(self):
        # Default traces and behaviors
        opt = dict(filter_mode='rolling_mean', interpolate_nan=True, nan_tracking_failure_points=True,
                   use_physical_time=True)

        self.df_traces = self.project_data.calc_default_traces(**opt)
        self.df_beh = self.project_data.calc_default_behaviors(**opt)
        # No filtering
        self.df_beh_binary = self.project_data.calc_default_behaviors(binary_behaviors=True)

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
            cca = scc_mod.SCCA_IPLS(latent_dims=n_components, tau=[1e-2, 1e-3])
            X_r, Y_r = cca.fit_transform([X, Y])

        return X_r, Y_r, cca

    def calc_cca_reconstruction(self, **kwargs):
        X_r, Y_r, cca = self.calc_cca(**kwargs)
        return cca.inverse_transform(X_r, Y_r)

    def calc_r_squared(self, use_pca=False, n_components=1, **kwargs):
        # First, calculate the reconstruction
        X = self.df_traces
        if use_pca:
            # See calc_pca_modes
            X = fill_nan_in_dataframe(X, do_filtering=False)
            X -= X.mean()
            pca = PCA(n_components=n_components, whiten=False)
            X_r_recon = pca.inverse_transform(pca.fit_transform(X))
        else:
            X_r_recon, Y_r_recon = self.calc_cca_reconstruction(n_components=n_components, **kwargs)
        # Then, calculate the r-squared
        residual_variance = (X - X_r_recon).var().sum()
        total_variance = X.var().sum()

        return 1 - residual_variance / total_variance

    def _get_beh_df(self, binary_behaviors):
        if binary_behaviors:
            Y = self.df_beh_binary
        else:
            Y = self.df_beh
        return Y

    def visualize_modes(self, i=1, binary_behaviors=False, **kwargs):
        X_r, Y_r, cca = self.calc_cca(n_components=i, binary_behaviors=binary_behaviors, **kwargs)

        df = pd.DataFrame({'Latent X': X_r[:, i], 'Latent Y': Y_r[:, i]})
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.show()

        return fig

    def visualize_modes_and_weights(self, n_components=1, binary_behaviors=False, **kwargs):

        X_r, Y_r, cca = self.calc_cca(n_components=n_components, binary_behaviors=binary_behaviors, **kwargs)
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces

        if 'sparse_tau' in kwargs:
            df_y = pd.DataFrame(cca.weights[1], index=df_beh.columns).T
            df_x = pd.DataFrame(cca.weights[0], index=df_traces.columns).T
        else:
            df_y = pd.DataFrame(cca.y_weights_, index=df_beh.columns).T
            df_x = pd.DataFrame(cca.x_weights_, index=df_traces.columns).T

        def f(i=0):
            df = pd.DataFrame({'Latent X': X_r[:, i] / X_r[:, i].max(),
                               'Latent Y': Y_r[:, i] / Y_r[:, i].max()})
            fig = px.line(df)
            self.project_data.shade_axis_using_behavior(plotly_fig=fig)
            fig.show()

            fig = px.bar(df_y.iloc[i, :])
            fig.show()

            fig = px.bar(df_x.iloc[i, :])
            fig.show()

        interact(f, i=(0, X_r.shape[1] - 1))

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
                scene_camera=dict(eye=dict(x=2.0, y=2.0, z=2.0))
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


def calc_r_squared_for_all_projects(all_projects, **r_squared_kwargs):
    all_cca_classes = {}
    all_r_squared = defaultdict(dict)

    opt_dict = {'pca': dict(use_pca=True),
                'cca': dict(use_pca=False),
                'cca_binary': dict(use_pca=False, binary_behaviors=True)}

    for name, p in all_projects.items():
        cca_plotter = CCAPlotter(p)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            all_r_squared[name][opt_name] = cca_plotter.calc_r_squared(**opt, **r_squared_kwargs)

    df_r_squared = pd.DataFrame(all_r_squared).T

    return all_cca_classes, df_r_squared
