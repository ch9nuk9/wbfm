#!/usr/bin/env python
# coding: utf-8




get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
from wbfm.utils.projects.finished_project_data import ProjectData
import napari
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import zarr
from pathlib import Path
import os
import seaborn as sns


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
import plotly.express as px


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# # Do PCA, CCA on real behaviors, and CCA on binarized behaviors

output_folder = 'cca'


from wbfm.utils.visualization.utils_cca import CCAPlotter


project_data_gcamp.use_physical_x_axis = True

cca_plotter = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True))


# fig = cca_plotter.plot(output_folder=output_folder, show_legend=False)


# For fixing cutoff labels
# camera = dict(
#     eye=dict(x=2, y=2, z=2))
# fig.update_layout(scene_camera=camera)
# fig.show()


# fig = cca_plotter.plot(binary_behaviors=True, output_folder=output_folder, show_legend=False)#, beh_annotation_kwargs=dict(include_collision=True))


# fig = cca_plotter.plot(binary_behaviors=False, use_pca=True, output_folder=output_folder, show_legend=False)


# ## Also plot correlation matrix for behaviors

# df_beh = cca_plotter._df_beh
# fig = px.imshow(df_beh.corr(), width=1000, height=1000)
# fig.show()





# ## Example dataset modes in 2d

fig = cca_plotter.plot(plot_3d=False, output_folder=output_folder, show_legend=False)


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=True, show_legend=False, output_folder=output_folder)#, beh_annotation_kwargs=dict(include_collision=True))


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=False, show_legend=False, use_pca=True, output_folder=output_folder)





# ## Example traces: Top mode

fig = cca_plotter.plot_single_mode(output_folder=output_folder, show_legend=False)


fig = cca_plotter.plot_single_mode(binary_behaviors=True, output_folder=output_folder, show_legend=False)#, beh_annotation_kwargs=dict(include_collision=True))


fig = cca_plotter.plot_single_mode(binary_behaviors=False, use_pca=True, output_folder=output_folder, show_legend=False)








# # Calculate variance explained per dataset

from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


all_cca_classes, df_r_squared = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=1), 
                                                                preprocess_traces_using_pca=True, truncate_traces_to_n_components=5, trace_kwargs=dict(use_paper_options=True))


# df_r_squared = df_r_squared.rename(columns={'CCA Discrete': 'CCA\n Discrete'})


df_r_squared_melt = df_r_squared.melt(var_name='Model Type', value_name='$R^2$')
# df_r_squared_melt


fig = px.box(df_r_squared_melt, x='Model Type', y='$R^2$', 
             color='Model Type', color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")

apply_figure_settings(fig, width_factor=0.25, height_factor=0.2)
# fig.update_yaxes(title='$R^2$', range=[0, 0.7])
fig.update_xaxes(title='')
fig.update_layout(showlegend=False)

fname = os.path.join(output_folder, 'top_mode_reconstruction_boxplot.png')
fig.write_image(fname, scale=3)

fig.show()





# fig = paired_boxplot_from_dataframes(df_r_squared.T, num_rows=3, add_median_line=False)
# plt.ylabel("R2")
# plt.title("Reconstruction (mode 1)")

# apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

# fname = os.path.join(output_folder, 'paired_boxplot.png')
# plt.savefig(fname, transparent=True)
# fname = Path(fname).with_suffix('.svg')
# plt.savefig(fname)


# ## Also do ttests

df_r_squared.head()


from scipy.stats import ttest_ind
from itertools import combinations


cols = list(df_r_squared.columns)
col_pairs = combinations(cols, 2)

for col_pair in col_pairs:
    a = df_r_squared.loc[:, col_pair[0]]
    b = df_r_squared.loc[:, col_pair[1]]

    result = ttest_ind(a, b, equal_var=False)
    print(f"Ttest for correlations of columns {col_pair}: {result.pvalue}")





# # Calculate dot product between the pca and cca modes

all_dots = {i+1: {name: c.calc_mode_dot_product(i) for name, c in tqdm(all_cca_classes.items())} for i in range(3)}
df_all_dots = pd.DataFrame(all_dots)


# %debug



fig = px.box(df_all_dots.abs())
fig.update_traces(marker=dict(color=plotly_paper_color_discrete_map()['PCA']))
fig.update_xaxes(title='Component')
fig.update_yaxes(title='PCA-CCA similarity', range=[0, 1.01])

apply_figure_settings(fig, width_factor=0.25, height_factor=0.2)

fig.show()

fname = os.path.join(output_folder, 'mode_dot_product.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## For Manuel: also do 3 modes

# all_cca_classes3, df_r_squared3 = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=3), 
#                                                                 preprocess_traces_using_pca=True, truncate_traces_to_n_components=5)


# fig = paired_boxplot_from_dataframes(df_r_squared3.T, num_rows=3, add_median_line=False)
# plt.ylabel("R squared")
# plt.title("Reconstruction quality for 3 modes")

# fname = os.path.join(output_folder, 'paired_boxplot_3components.png')
# plt.savefig(fname)





# # Look at the correlation of the modes across datasets

from wbfm.utils.visualization.utils_cca import calc_mode_correlation_for_all_projects
from wbfm.utils.general.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.general.utils_paper import apply_figure_settings


output_folder = 'cca'


all_cca_classes3, df_mode_correlations, df_mode_correlations_binary = calc_mode_correlation_for_all_projects(all_projects_gcamp, correlation_kwargs=dict(n_components=5),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=6, 
                                                                                                             trace_kwargs=dict(use_paper_options=True))


df_mode_correlations.index = np.arange(1, 6)
df_mode_correlations_binary.index = np.arange(1, 6)


fig = paired_boxplot_from_dataframes(df_mode_correlations, num_rows=5, add_median_line=False)
plt.ylabel("Correlation")
plt.xlabel("CCA mode index")
plt.title("CCA")

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

fname = os.path.join(output_folder, 'paired_boxplot_latent_space.png')
plt.savefig(fname, transparent=True)
fname = Path(fname).with_suffix('.svg')
plt.savefig(fname)


fig = paired_boxplot_from_dataframes(df_mode_correlations_binary, num_rows=5, add_median_line=False)
plt.ylabel("Correlation")
plt.xlabel("CCA mode index")
plt.title("CCA (binary)")

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

fname = os.path.join(output_folder, 'paired_boxplot_latent_space_binary.png')
plt.savefig(fname, transparent=True)
fname = Path(fname).with_suffix('.svg')
plt.savefig(fname)


df0 = df_mode_correlations.T.copy()
df1 = df_mode_correlations_binary.T.copy()

df0['Behavior type'] = 'Continuous'
df1['Behavior type'] = 'Discrete'
df_mode_combined = pd.concat([df0, df1])


fig = px.box(df_mode_combined, color='Behavior type')

fig.update_xaxes(title="CCA Mode index")
fig.update_yaxes(title="Correlation <br> of latent space")

apply_figure_settings(fig, width_factor=0.6, height_factor=0.2)
fig.show()


fname = os.path.join(output_folder, 'paired_boxplot_latent_space_combined.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Also do t-tests

from scipy.stats import ttest_ind


df_mode_correlations


a = df_mode_correlations.loc[0, :]
b = df_mode_correlations_binary.loc[0, :]

result = ttest_ind(a, b, equal_var=False)
print(f"Ttest for correlations of mode 0: {result.pvalue}")


a = df_mode_correlations.loc[1, :]
b = df_mode_correlations_binary.loc[1, :]

result = ttest_ind(a, b, equal_var=False)
print(f"Ttest for correlations of mode 1: {result.pvalue}")


# # Get the neural and behavioral weights across datasets

from wbfm.utils.visualization.utils_cca import calc_cca_weights_for_all_projects
import plotly.express as px
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
output_folder = 'cca'


all_cca_classes1, df_weights1, df_weights_binary1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=2),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# df_weights1 = df_weights1[[c for c in df_weights1.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights1)#, title="CCA weights of mode 1 across recordings")

# apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
# # fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 1)")
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight")
# fig.update_xaxes(title="")
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join(output_folder, 'paired_boxplot_neural_weights.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# df_weights_binary1 = df_weights_binary1[[c for c in df_weights_binary1.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights_binary1)#, title="CCA weights of mode 1 across recordings (binary)")

# apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 1)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# Both modes together
df_both1 = df_weights1.reset_index().melt(id_vars='index')
df_both1['Behavior Type'] = 'Continuous'
df_both1_binary = df_weights_binary1.reset_index().melt(id_vars='index')
df_both1_binary['Behavior Type'] = 'Discrete'
df_both1 = pd.concat([df_both1, df_both1_binary])
df_both1.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
# df_both1


fig = px.box(df_both1, x='Neuron', y='Weight', color='Behavior Type',
             color_discrete_map=plotly_paper_color_discrete_map())

apply_figure_settings(fig, width_factor=0.7, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight")
fig.update_xaxes(title="")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.5
))
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_BOTH.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## SUPP: Same but for mode 2

all_cca_classes2, df_weights2, df_weights_binary2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


df_weights2 = df_weights2[[c for c in df_weights2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

fig = px.box(df_weights2, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])

apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 2)")
fig.update_xaxes(title="")
fig.show()

to_save = True
if to_save:
    fname = os.path.join(output_folder, 'paired_boxplot_neural_weights2.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


df_weights_binary2 = df_weights_binary2[[c for c in df_weights_binary2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

fig = px.box(df_weights_binary2, color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])#, title="CCA weights of mode 2 across recordings (binary)")

apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## SUPP: Same but for mode 3

all_cca_classes3, df_weights3, df_weights_binary3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


df_weights3 = df_weights3[[c for c in df_weights3.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

fig = px.box(df_weights3, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])#, title="CCA weights of mode 3 across recordings")
apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights3.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# fig = px.box(df_weights_binary, title="CCA weights of mode 3 across recordings (binary)")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary3.png')
# fig.write_image(fname)


# ## Same but for behavior weights

from wbfm.utils.general.utils_paper import behavior_name_mapping


all_cca_classes_beh1, df_weights_beh1, df_weights_binary_beh1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=2), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


fig = px.box(df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight")
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight  <br> (mode 1)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


cmap = px.colors.qualitative.Plotly
# df_weights_binary_beh1['color'] = cmap[1]

fig = px.box(df_weights_binary_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Supp: behavior

all_cca_classes_beh2, df_weights_beh2, df_weights_binary_beh2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


fig = px.box(df_weights_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


fig = px.box(df_weights_binary_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])

apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


all_cca_classes_beh3, df_weights_beh3, df_weights_binary_beh3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


fig = px.box(df_weights_beh3.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights3.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)








# # Scratch: Color code the CCA space

# behavior_list = ['signed_speed_angular', 'ventral_only_body_curvature', 'ventral_only_head_curvature', 'dorsal_only_body_curvature', 'dorsal_only_head_curvature']
# for beh_name in behavior_list:
#     fig = cca_plotter.plot(plot_3d=True, binary_behaviors=False, show_legend=False, output_folder='cca/exploration', overwrite_file=False, use_paper_options=False,
#                           color_by_discrete_behavior=False, color_by_continuous_behavior=beh_name)














# # Scratch: Make sure mode 1 is actually the FWD/REV mode

# for name, obj in all_cca_classes.items():
#     print(name)
#     obj.plot_single_mode()
#     # obj.plot_single_mode(1)
#     # obj.plot_single_mode(2)
#     # obj.plot_single_mode(binary_behaviors=True)











# # Scratch: Visualize modes

from wbfm.utils.visualization.utils_cca import CCAPlotter


cca_plotter3 = CCAPlotter(project_data_gcamp, preprocess_traces_using_pca=True, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True)


fig = cca_plotter3.visualize_modes_and_weights(binary_behaviors=False, n_components=5)


cca_plotter4 = CCAPlotter(project_data_gcamp, preprocess_traces_using_pca=False, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True)


fig = cca_plotter4.visualize_modes_and_weights(binary_behaviors=False, n_components=5)


# fig = cca_plotter.plot()


# # Scratch: Visualize across all datasets

all_cca_classes, df_r_squared = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=1), 
                                                                preprocess_traces_using_pca=True, truncate_traces_to_n_components=None)


# # Same GUI but for all datasets
# from ipywidgets import interact

# def f(dataset_name):
#     cca_plotter = all_cca_classes[dataset_name]
#     fig = cca_plotter.visualize_modes_and_weights(binary_behaviors=False, n_components=8)
    
# interact(f, dataset_name=list(all_cca_classes.keys()))


all_cca_classes2, df_r_squared2 = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=1), 
                                                                preprocess_traces_using_pca=False, preprocess_behavior_using_pca=False)


# # Same GUI but for all datasets
# from ipywidgets import interact

# def f(dataset_name):
#     cca_plotter = all_cca_classes2[dataset_name]
#     fig = cca_plotter.visualize_modes_and_weights(binary_behaviors=False, n_components=8, sparse_tau=(1e-2, 1e-2))
    
# interact(f, dataset_name=list(all_cca_classes.keys()))


# # Scratch: look at the angle of rotation 

from wbfm.utils.visualization.utils_cca import CCAPlotter
from scipy.spatial.transform import Rotation


cca_plotter3 = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=3, preprocess_behavior_using_pca=False)


X_r, Y_r, cca = cca_plotter3.calc_cca(n_components=1)


# r = Rotation.from_matrix(cca.x_rotations_)


cca.x_weights_, cca.x_loadings_








# # Debug






