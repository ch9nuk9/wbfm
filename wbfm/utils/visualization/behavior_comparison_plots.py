import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from wbfm.utils.projects.finished_project_data import ProjectData


def plot_correlation_of_examples(project_path):
    project_data = ProjectData.load_final_project_data_from_config(project_path)

    opt = dict(channel_mode='red')
    red = project_data.calc_default_traces(**opt)
    opt = dict(channel_mode='green')
    green = project_data.calc_default_traces(**opt)
    opt = dict(channel_mode='ratio')
    ratio = project_data.calc_default_traces(**opt)

    opt = dict(channel_mode='ratio_rolling_ransac')
    ratio_ransac = project_data.calc_default_traces(**opt)
    opt = dict(channel_mode='ratio', filter_mode='rolling_mean')
    ratio_filt = project_data.calc_default_traces(**opt)
    # Align columns
    to_drop = set(ratio_filt.columns) - set(red.columns)
    ratio_filt.drop(columns=to_drop, inplace=True)

    all_dfs = [red, green, ratio, ratio_ransac, ratio_filt]
    all_labels = ['red', 'green', 'ratio', 'ratio_ransac', 'ratio_filt']

    # Calculate correlation dataframes
    kymo = project_data.worm_posture_class.curvature_fluorescence_fps.reset_index(drop=True, inplace=False)
    all_dfs_corr = calculate_correlations_traces_and_kymo(kymo, all_dfs)


def calculate_correlations_traces_and_kymo(kymo, all_dfs):
    kymo_smaller = kymo.loc[:, 3:30].copy()

    all_dfs_corr = [np.abs(pd.concat([df, kymo_smaller], axis=1).corr()) for df in all_dfs]

    # Only get the corner we care about: traces vs. kymo
    ind_nonneuron = np.arange(all_dfs[0].shape[1], all_dfs_corr[0].shape[1])
    ind_neurons = np.arange(0, all_dfs[0].shape[1])
    all_dfs_corr = [df.iloc[ind_neurons, ind_nonneuron] for df in all_dfs_corr]

    return all_dfs_corr


def _multi_plot(all_dfs, all_dfs_corr, all_labels, all_colors=None, ax_locations=None,
                project_data: ProjectData=None,
                corr_thresh=0.3, which_df_to_apply_corr_thresh=0, num_plots=None,
                xlim=None, to_save=False):
    if xlim is None:
        xlim = [100, 450]
    if ax_locations is None:
        ax_locations = [1, 1, 3, 3, 3]
    if all_colors is None:
        all_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange']
    if num_plots is None:
        num_plots = all_dfs_corr[0].shape[0]

    for i in range(num_plots):
        corr = np.abs(all_dfs_corr[which_df_to_apply_corr_thresh].iloc[i, :])
        if corr.max() < corr_thresh:
            continue

        fig, axes = plt.subplots(ncols=2, nrows=2, dpi=100, figsize=(15, 5))
        axes = np.ravel(axes)
        neuron_name = all_dfs_corr[0].columns[i]

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
        axes[0].set_title(neuron_name)
        axes[0].legend()

        axes[1].set_xlim(xlim[0], xlim[1])
        axes[3].set_xlim(xlim[0], xlim[1])
        axes[1].legend()
        axes[3].legend()
        axes[3].set_xlabel("Time (frames)")
        axes[1].set_ylabel("Normalized amplitude")
        if project_data is not None:
            project_data.shade_axis_using_behavior(axes[1])
            project_data.shade_axis_using_behavior(axes[3])

        axes[2].remove()

        if to_save:
            vis_cfg = project_data.project_config.get_visualization_config()
            fname = f'traces_kymo_correlation_{neuron_name}.png'
            fname = vis_cfg.resolve_relative_path(vis_cfg, prepend_subfolder=True)

            plt.savefig(fname)

        # plt.show()
