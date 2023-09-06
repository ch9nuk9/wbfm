import os
from dataclasses import dataclass
import pandas as pd
from ipywidgets import interact
from sklearn.cross_decomposition import CCA
import plotly.express as px
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
        opt = dict(filter_mode='rolling_mean', interpolate_nan=True, nan_tracking_failure_points=True)

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
            cca = scc_mod.SCCA_IPLS(latent_dims=4, tau=[1e-2, 1e-3])
            X_r, Y_r = cca.fit_transform([X, Y])

        return X_r, Y_r, cca

    def _get_beh_df(self, binary_behaviors):
        if binary_behaviors:
            Y = self.df_beh_binary
        else:
            Y = self.df_beh
        return Y

    def visualize_modes(self, i=1, binary_behaviors=False, **kwargs):
        X_r, Y_r, cca = self.calc_cca(n_components=i, binary_behaviors=binary_behaviors, **kwargs)
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces

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

    def plot_3d(self, binary_behaviors=False, modes_to_plot=None, use_pca=False, use_X_r=True, sparse_tau=None,
                output_folder=None, DEBUG=False,
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
                              include_head_cast=False, include_pause=False, include_hesitation=False)
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
            phase_plot_list.append(
                go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                             mode='lines',
                             name=state_code.full_name, line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

        fig = go.Figure(layout=dict(height=1000, width=1000))
        fig.add_traces(phase_plot_list)

        # Hacky: https://community.plotly.com/t/scatter3d-background-plot-color/38838/4
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    tickvals=[-1, 0, 1],
                    showbackground=True,
                    gridcolor='black',
                    zerolinecolor="white",
                    title='Mode 1'
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    tickvals=[-1, 0, 1],
                    showbackground=True,
                    gridcolor='black',
                    zerolinecolor="white",
                    title='Mode 2'),
                zaxis=dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    tickvals=[-1, 0, 1],
                    showbackground=True,
                    gridcolor='black',
                    zerolinecolor="white",
                    title='Mode 3'),
            ),
            # From: https://stackoverflow.com/questions/73187799/truncated-figure-with-plotly?noredirect=1#comment129258910_73187799
            scene_camera=dict(eye=dict(x=2.0, y=2.0, z=0.75))
        )

        if output_folder is not None:
            # Build name based on options used
            if use_pca:
                fname = 'pca_3d.html'
            else:
                if binary_behaviors:
                    fname = 'cca_binary_3d.html'
                else:
                    fname = 'cca_continuous_3d.html'
            fname = os.path.join(output_folder, fname)
            fig.write_html(fname)
            fname = fname.replace('.html', '.png')
            fig.write_image(fname)

        fig.show()
        return fig
