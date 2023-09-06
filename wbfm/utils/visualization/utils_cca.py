from dataclasses import dataclass
import pandas as pd
from ipywidgets import interact
from sklearn.cross_decomposition import CCA
import plotly.express as px
from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
import plotly.graph_objects as go
from methodtools import lru_cache

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
        opt = dict(filter_mode='rolling_mean', interpolate_nan=True)

        self.df_traces = self.project_data.calc_default_traces(**opt)
        self.df_beh = self.project_data.calc_default_behaviors(**opt)
        # No filtering
        self.df_beh_binary = self.project_data.calc_default_behaviors(binary_behaviors=True)

    @lru_cache(maxsize=4)
    def calc_cca(self, n_components, binary_behaviors):

        X = self.df_traces
        Y = self._get_beh_df(binary_behaviors)

        cca = CCA(n_components=n_components)
        X_r, Y_r = cca.fit_transform(X, Y)

        Y_r_linear = self.df_traces @ cca.coef_

        return X_r, Y_r, Y_r_linear, cca

    def _get_beh_df(self, binary_behaviors):
        if binary_behaviors:
            Y = self.df_beh_binary
        else:
            Y = self.df_beh
        return Y

    def visualize_modes(self, i=1, binary_behaviors=False):
        X_r, Y_r, Y_r_linear, cca = self.calc_cca(n_components=i, binary_behaviors=binary_behaviors)
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces

        df = pd.DataFrame({'Latent X': X_r[:, i], 'Latent Y': Y_r[:, i]})
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.show()

        return fig

    def visualize_modes_and_weights(self, n_components=1, binary_behaviors=False):

        X_r, Y_r, Y_r_linear, cca = self.calc_cca(n_components=n_components, binary_behaviors=False)
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces

        df_y = pd.DataFrame(cca.y_weights_, index=df_beh.columns).T
        df_x = pd.DataFrame(cca.x_weights_, index=df_traces.columns).T

        def f(i=0):
            df = pd.DataFrame({'Latent X': X_r[:, i], 'Latent Y': Y_r[:, i]})
            fig = px.line(df)
            self.project_data.shade_axis_using_behavior(plotly_fig=fig)
            fig.show()

            fig = px.bar(df_y.iloc[i, :])
            fig.show()

            fig = px.bar(df_x.iloc[i, :])
            fig.show()

        interact(f, i=(0, X_r.shape[1] - 1))

    def plot_3d(self, binary_behaviors=False, modes_to_plot=None, use_pca=False, **ethogram_cmap_kwargs):
        if modes_to_plot is None:
            modes_to_plot = [0, 1, 2]
        if use_pca:
            X_r = self.project_data.calc_pca_modes(n_components=3)
        else:
            X_r, Y_r, Y_r_linear, cca = self.calc_cca(n_components=3, binary_behaviors=binary_behaviors)

        df_latents = pd.DataFrame(X_r)
        df_latents['state'] = self.project_data.worm_posture_class.beh_annotation(fluorescence_fps=True,
                                                                                  reset_index=True,
                                                                                  include_collision=False,
                                                                                  include_turns=True,
                                                                                  include_head_cast=False,
                                                                                  include_pause=False,
                                                                                  include_hesitation=False)
        ethogram_cmap_kwargs.setdefault('include_turns', True)
        ethogram_cmap_kwargs.setdefault('include_quiescence', False)
        ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs)
        df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_latents, modes_to_plot, 'state')
        state_codes = df_latents['state'].unique()

        phase_plot_list = []
        for i, state_code in enumerate(state_codes):
            phase_plot_list.append(
                go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                             mode='lines',
                             name=state_code.full_name, line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

        fig = go.Figure(layout=dict(height=1000, width=1000))
        fig.add_traces(phase_plot_list)
        fig.show()

        return fig
