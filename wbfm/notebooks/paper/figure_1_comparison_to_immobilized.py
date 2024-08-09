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
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
import plotly.express as px
from wbfm.utils.general.utils_filenames import add_name_suffix


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


all_projects_gfp = load_paper_datasets('gfp')


all_projects_immob = load_paper_datasets('immob')


# Get specific example datasets
project_data_gcamp = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022_11_28']
project_data_immob = all_projects_immob['2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']


# Same individual: fm and immob
fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-23_ZIM2165_worm5-2022-12-06/project_config.yaml'
project_data_fm2immob_fm = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-41_ZIM2165_immob_worm5-2022-12-06'
project_data_fm2immob_immob = ProjectData.load_final_project_data_from_config(fname, verbose=0)


# project_data_gcamp


path_to_saved_data = "../step1_analysis/figure_1"
path_to_shared_saved_data = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/step1_analysis/shared"


# # Precalculate the trace dataframes (cached to disk)

# # Optional: clear just one cache
# for project_dict in tqdm([all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# # Optional: clear the trace cache
# for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, 
#                           all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, all_projects_immob]):
    for name, project_data in tqdm(project_dict.items()):
        df_traces = project_data.calc_paper_traces()
        df_res = project_data.calc_paper_traces(residual_mode='pca')
        df_global = project_data.calc_paper_traces_global(residual_mode='pca_global')
        if df_res is None or df_global is None or df_traces is None:
            raise ValueError


project_data_fm2immob_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# Also for FM to IMMOB datasets

for project_data in [project_data_fm2immob_fm, project_data_fm2immob_immob]:
    df_traces = project_data.calc_paper_traces()
    df_res = project_data.calc_paper_traces(residual_mode='pca')
    df_global = project_data.calc_paper_traces(residual_mode='pca_global')
    if df_res is None or df_global is None or df_traces is None:
        raise ValueError


# # Plots

# ## Heatmaps: immobilized and WBFM

from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


project_data_gcamp.use_physical_x_axis = True
project_data_immob.use_physical_x_axis = True


fig = make_summary_interactive_heatmap_with_pca(project_data_gcamp, to_save=True, to_show=True, output_folder="intro/example_summary_plots_wbfm")


# fig = make_summary_interactive_heatmap_with_pca(project_data_immob, to_save=True, to_show=True, output_folder="example_summary_plots_immob")


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp, trace_opt=dict(use_paper_options=True), to_save=True, to_show=True, 
                                               output_folder="intro/example_summary_plots_wbfm")


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_immob, trace_opt=dict(use_paper_options=True), include_speed_subplot=False,
                                               to_save=True, to_show=True, output_folder="intro/example_summary_plots_immob")


# ## Heatmaps: same dataset fm to immob

from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


project_data_fm2immob_fm.use_physical_x_axis = True
project_data_fm2immob_immob.use_physical_x_axis = True


# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
# approximate_behavioral_annotation_using_pc1(project_data_fm2immob_immob)
# approximate_behavioral_annotation_using_pc1(project_data_fm2immob_fm)  # This is an old dataset and the behavior was deleted


project_data_fm2immob_immob


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_fm2immob_immob, trace_opt=dict(use_paper_options=True, interpolate_nan=False, verbose=True), include_speed_subplot=False,
                                               to_save=True, to_show=True, output_folder="intro/fm_to_immob/immob")


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_fm2immob_fm, trace_opt=dict(use_paper_options=True, interpolate_nan=False, verbose=True), include_speed_subplot=False,
                                               to_save=True, to_show=True, output_folder="intro/fm_to_immob/fm")


trace_opt=dict(use_paper_options=True, interpolate_nan=False, verbose=True)
df = project_data_fm2immob_immob.calc_default_traces(**trace_opt)


# df['neuron_014']


# project_data_fm2immob_immob.tail_neuron_names()





# ## Triggered average examples

# from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, shade_using_behavior, shade_triggered_average
from wbfm.utils.visualization.utils_plot_traces import plot_triggered_averages





plot_triggered_averages([project_data_gcamp, project_data_immob], output_foldername="intro/basic_triggered_average")


# ## PCA variance explained plot of all datasets

from wbfm.utils.visualization.multiproject_wrappers import get_all_variance_explained
from wbfm.utils.visualization.utils_plot_traces import plot_with_shading
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


gcamp_var, gfp_var, immob_var, gcamp_var_sum, gfp_var_sum, immob_var_sum = get_all_variance_explained(all_projects_gcamp, all_projects_gfp, all_projects_immob)


fig, ax = plt.subplots(dpi=200, figsize=(5,5))

var_sum_dict = {'Freely Moving (GCaMP)': gcamp_var_sum, 'Immobilized (GCaMP)': immob_var_sum, 'Freely Moving (GFP)': gfp_var_sum}
cmap = plotly_paper_color_discrete_map()

for name, mat in var_sum_dict.items():
    means = np.mean(mat, axis=1)
    color = cmap[name]
    plot_with_shading(means, np.std(mat, axis=1), label=name, ax=ax, lw=2,
                      x=np.arange(1, len(means) + 1), color=color)
# plt.legend()
# plt.title("Dimensionality")
plt.ylabel("Cumulative explained variance")
plt.ylim(0.2, 1.0)
plt.xlabel("Mode")

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

apply_figure_settings(fig, width_factor=0.2, height_factor=0.25, plotly_not_matplotlib=False)
plt.tight_layout()

output_foldername = 'intro'
fname = f"pca_cumulative_variance.png"
fname = os.path.join(output_foldername, fname)
plt.savefig(fname, transparent=True)
fig.savefig(fname.replace(".png", ".svg"), transparent=True)








# # PCA weights across wbfm and immob

from wbfm.utils.visualization.utils_cca import calc_pca_weights_for_all_projects
from wbfm.utils.external.utils_plotly import plotly_boxplot_colored_boxes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids


neuron_names = neurons_with_confident_ids()



wbfm_weights = calc_pca_weights_for_all_projects(all_projects_gcamp, use_paper_options=True, combine_left_right=True,
                                                neuron_names=neuron_names)


immob_weights = calc_pca_weights_for_all_projects(all_projects_immob, use_paper_options=True, combine_left_right=True,
                                                 neuron_names=neuron_names)


# # Create a list of colors to highlight BAG
# base_color = '#1F77B4'  # Gray
# names = list(wbfm_weights.columns)
# colors = ['#000000' if 'BAG' in n else base_color for n in names]

# # fig = px.box(wbfm_weights)
# fig = plotly_boxplot_colored_boxes(wbfm_weights, colors)
# # Transparent background
# apply_figure_settings(fig, width_factor=0.6, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black")
# fig.update_xaxes(dict(title=""))
# # Add the 0 line back

# fig.show()

# fname = os.path.join("intro", 'wbfm_pca_weights.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# # Create a list of colors to highlight BAG
# base_color = '#FF7F0E'  # Orange, overall immob color
# names = list(immob_weights.columns)
# # colors = ['#EF553B' if 'BAG' in n else base_color for n in names]
# colors = ['#000000' if 'BAG' in n else base_color for n in names]

# fig = plotly_boxplot_colored_boxes(immob_weights, colors)
# apply_figure_settings(fig, width_factor=0.6, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black")
# fig.update_yaxes(dict(title="PC1 weight"))
# fig.update_xaxes(dict(title="Neuron Name"))
# fig.show()

# fname = os.path.join("intro", 'immob_pca_weights.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# ## FM and immob on same plot

from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
from wbfm.utils.general.utils_paper import data_type_name_mapping, plotly_paper_color_discrete_map
import plotly.graph_objects as go


names_to_keep = set(wbfm_weights.columns).intersection(immob_weights.columns)
wbfm_melt = wbfm_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='gcamp')
immob_melt = immob_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='immob')
df_both = pd.concat([wbfm_melt, immob_melt], axis=0)
df_both = df_both[df_both['neuron_name'].isin(names_to_keep)]
df_both['Dataset Type'] = df_both['dataset_type'].map(data_type_name_mapping())
df_both['neuron_name'].unique()



fig = px.box(df_both, y='PC1 weight', x='neuron_name', color='Dataset Type', 
            color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})

add_p_value_annotation(fig, x_label='all', show_ns=False, show_only_stars=True, permutations=1000,
                      height_mode='top_of_data')#, _format=dict(text_height=0.075))
apply_figure_settings(fig, width_factor=0.7, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.8,
    xanchor="left",
    x=0.6
))
# Add a fake trace to get a legend entry for GFP
dummy_data = go.Box(
    x=[None],
    y=[None],
    # mode="markers",
    name="Freely Moving (GFP)",
    #fillcolor='gray', 
    line=dict(color='gray')
    # marker=dict(size=7, color="gray"),
)
fig.add_trace(dummy_data)

fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black")
fig.update_xaxes(dict(title="Neuron Name"))
fig.show()

fname = os.path.join("intro", 'fm_and_immob_pca_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Pie chart summarizing above
# 
# Get the p values between immob and fm and build 4 categories:
# 1. No difference, both positive
# 2. No difference, both negative
# 3. Positive difference
# 4. Negative difference
# 
# ... problem: AVA will then show a very strong difference, as will many that stay "on the same side"... what we really want is a switch in correlations for category 3/4
# New categories:
# 3. immob NS diff from 0, fm positive
# 4. immob NS diff from 0, fm negative
# 5. Both NS diff from 0, but not from each other (will this exist?)
# 6. fm NS diff from 0, immob positive
# 7. fm NS diff from 0, immob negative
# 8. Both NS diff from 0
# 
# ... this is a lot of categories! I can maybe just ignore categories 6-8, as they are not interesting in FM

df_both.head()


from scipy import stats
from statsmodels.stats.multitest import multipletests

# 
func = lambda x: stats.ttest_1samp(x, 0)[1]
df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
df_pvalue = df_groupby['PC1 weight'].apply(func).to_frame()
df_pvalue.columns = ['p_value']
output = multipletests(df_pvalue.values.squeeze(), method='bonferroni')
df_pvalue['p_value_corrected'] = output[1]
df_pvalue['significance_corrected'] = output[0]

df_medians = df_groupby['PC1 weight'].median()[(slice(None), 'gcamp')]
# df_medians_diff = df_medians[(slice(None), 'gcamp')] - df_medians[(slice(None), 'immob')]



# # Add horizontal line at the intersection with mean_y

# fig = px.scatter(df_pvalue.reset_index(), x='neuron_name', y='p_value_corrected', color='dataset_type',
#                  symbol = 'significance_corrected',
#           log_y=True)
# fig.add_shape(type="line",
#               x0=0, y0=0.05, x1=1, y1=0.05, xref='paper',
#               line=dict(color="Black", width=1, dash="dash"),
#               )
# fig.show()


# df_pvalue_thresh = (df_pvalue*len(df_pvalue) < 0.05).reset_index()
df_pvalue_thresh = df_pvalue['significance_corrected'].reset_index()

df_pivot = df_pvalue_thresh.pivot_table(index='neuron_name', columns='dataset_type', values='significance_corrected', aggfunc='first')
df_4states = df_pivot.astype(str).radd(df_pivot.columns + '_')
df_4states = (df_4states['gcamp'] + '_' + df_4states['immob'])#.reset_index()

df_medians.name = 'pc1_diff'
df_medians_diff_str = (df_medians > 0).astype(str).radd(df_medians.name + '_')
df_4states = df_4states.to_frame().join(df_medians_diff_str)#.reset_index()

# Combine columns again
df_4states.columns = ['pvalue_result', 'diff_result']
df_4states = (df_4states['pvalue_result'] + '_' + df_4states['diff_result']).to_frame()
df_4states.columns = ['Result']


# df_4states.sort_values(by='Result')


# df_4states['Result'].value_counts().reset_index()


df_4states_counts = df_4states['Result'].value_counts().reset_index()

# name_mapping = {
#     'gcamp_False_immob_False': 'No manifold',
#     'gcamp_False_immob_True': 'Reduce manifold',
#     'gcamp_True_immob_False': 'Increase manifold',
#     'gcamp_True_immob_True': 'Keep manifold'
# }
name_mapping = {
    'gcamp_True_immob_True_pc1_diff_True': 'Rev in both',
    'gcamp_True_immob_True_pc1_diff_False': 'Fwd in both',
    'gcamp_True_immob_False_pc1_diff_False': 'Fwd in FM only',
    'gcamp_False_immob_True_pc1_diff_False': 'Fwd in immob only',
    'gcamp_True_immob_False_pc1_diff_True': 'Rev in FM only',
    'gcamp_False_immob_False_pc1_diff_False': 'No manifold',
    'gcamp_False_immob_False_pc1_diff_True': 'No manifold',
    'gcamp_False_immob_True_pc1_diff_True': 'Rev in immob only',
}
df_4states_counts['Result'] = df_4states_counts['Result'].map(name_mapping)

def func(x):
    if 'both' in x:
        return 'Conserved'
    elif 'FM only' in x:
        return 'Freely moving only'
    elif 'immob only' in x:
        return 'Immobilized only'
    else:
        return 'No manifold participation'

df_4states_counts['Result_simple'] = df_4states_counts['Result'].map(func)

d2 = px.colors.qualitative.Dark2
s2 = px.colors.qualitative.Set2
cmap = {'Rev in both': d2[1],
       'Rev in FM only': s2[1],
       'Fwd in both': d2[0],
       'Fwd in FM only': s2[0],
       'No manifold': s2[-1],
       # 'Rev in immob only': s2[-1],
       # 'Fwd in immob only': s2[-1]
       }

d3 = px.colors.qualitative.D3
cmap = {'Rev in both': d3[4],
       'Rev in FM only': d3[0],
       'Fwd in both': d3[4],
       'Fwd in FM only': d3[0],
       'No manifold': d3[-3],
       'Rev in immob only': d3[2],
       'Fwd in immob only': d3[2]
       }

# Drop uninteresting rows
# df_4states_counts = df_4states_counts.drop(df_4states_counts[df_4states_counts['Result'].str.contains('No')].index)
# df_4states_counts = df_4states_counts.drop(df_4states_counts[df_4states_counts['Result'].str.contains('immob only')].index)

fig = px.pie(df_4states_counts, names='Result_simple', values='count', color='Result', color_discrete_map=cmap)
apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)
# fig.update_layout(legend=dict(
#     yanchor="top",
#     y=0.99,
#     xanchor="left",
#     x=0.01
# ))

# fig.update_traces(
#         textposition="outside",
#         texttemplate='%{percent:01f}')
fig.show()

output_foldername = 'intro'
fname = os.path.join(output_foldername, 'manifold_participation_pie_chart.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


df_4states_counts.drop(df_4states_counts[df_4states_counts['Result'].str.contains('No')].index)


# # Variance explained by mode 1 across neurons
# 
# i.e. the cumulative histogram, with error bar per dataset

from wbfm.utils.visualization.multiproject_wrappers import build_dataframe_of_variance_explained


trace_opt = dict(use_paper_options=True)
opt = dict(n_components=1, melt=True)

df_var_exp_gcamp = build_dataframe_of_variance_explained(all_projects_gcamp, **opt, **trace_opt)
df_var_exp_gcamp['Type of data'] = 'gcamp'
df_var_exp_immob = build_dataframe_of_variance_explained(all_projects_immob, **opt, **trace_opt)
df_var_exp_immob['Type of data'] = 'immob'
df_var_exp = pd.concat([df_var_exp_gcamp, df_var_exp_immob], axis=0)
df_var_exp.head()


# df_var_exp['neuron_name'].unique()


# px.histogram(df_var_exp, color='dataset_name', x='fraction_variance_explained', cumulative=True, 
#              facet_row='Type of data',
#              barmode='overlay', histnorm='percent')


df_var_exp_hist = df_var_exp.copy()

bins = np.linspace(0, 1, 50)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby('dataset_name')['fraction_variance_explained'].apply(func)
df_var_exp_hist.head()


# px.line(df_var_exp_hist['2022-11-23_worm10'])


long_vars = df_var_exp_hist.reset_index().explode('fraction_variance_explained')
long_vars.rename(columns={'fraction_variance_explained': 'cumulative_fraction_variance_explained'}, inplace=True)
long_vars.sort_values(by=['dataset_name', 'cumulative_fraction_variance_explained'], inplace=True)
# Just remake the bins
long_vars['cumcount'] = long_vars.groupby('dataset_name').cumcount()
long_vars['fraction_count'] = long_vars['cumcount'] / long_vars['cumcount'].max()

# Add back datatype column
long_vars = long_vars.merge(df_var_exp[['dataset_name', 'Type of data']], on='dataset_name')

# Normalize by number of total neurons
total_num_neurons = df_var_exp.dropna()['dataset_name'].value_counts()
long_vars.index = long_vars['dataset_name']  # So the division matches
long_vars['cumulative_fraction_variance_explained'] = long_vars['cumulative_fraction_variance_explained'] / total_num_neurons
long_vars.reset_index(drop=True, inplace=True)

long_vars.head()


# px.line(long_vars, x='fraction_count', 
#         y='cumulative_fraction_variance_explained', color='dataset_name',
#        facet_row='Type of data')


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading

opt = dict(x='fraction_count', y='cumulative_fraction_variance_explained', color='dataset_name', 
           cmap=plotly_paper_color_discrete_map())

fig = None
for g in ['gcamp']:#, 'immob']:
    fig = plotly_plot_mean_and_shading(long_vars[long_vars['Type of data']==g], line_name=g, fig=fig, **opt,
                                      x_intersection_annotation=0.5)

fig.update_xaxes(title='Variance explained <br> by PC1 (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(showlegend=False)
# fig.update_traces(line=dict(color=plotly_paper_color_discrete_map()['PCA']))

apply_figure_settings(fig, width_factor=0.25, height_factor=0.2)

fig.show()

output_foldername = 'intro/dimensionality'
fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# Add immob
g = 'immob'
fig = plotly_plot_mean_and_shading(long_vars[long_vars['Type of data']==g], line_name=g, fig=fig, **opt,
                                  x_intersection_annotation=0.5, annotation_position='right')

# In the supp, so it's larger
apply_figure_settings(fig, width_factor=0.5, height_factor=0.5)

fig.show()

output_foldername = 'intro/dimensionality'
fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative_with_immob.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Not used: gfp

# gfp_weights = calc_pca_weights_for_all_projects(all_projects_gfp)


# fig = px.box(gfp_weights)
# fig.show()

# fname = os.path.join(output_folder, 'gfp_pca_weights.png')
# fig.write_image(fname)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)























# # Scratch

# # Other

# ## Plateau in PC1 and speed during reversals



from wbfm.utils.external.utils_pandas import calc_eventwise_cooccurrence_matrix


# p = project_data_gcamp

df_all_occurances = []
for name, p in tqdm(all_projects_gcamp.items()):
    speed_plateau_state, _ = p.worm_posture_class.calc_piecewise_linear_plateau_state(replace_nan=True)
    pc1_plateau_state, _ = p.calc_plateau_state_using_pc1(replace_nan=True)

    rev_state = p.worm_posture_class.calc_behavior_from_alias('rev').astype(int)
    df = pd.DataFrame({'speed': speed_plateau_state, 'pc1': pc1_plateaus, 'rev': rev_state})
    df = df.fillna(False)

    df_occur = calc_eventwise_cooccurrence_matrix(df, 'speed', 'pc1', 'rev')
    df_occur['dataset'] = name
    df_all_occurances.append(df_occur)
df_all_cooccurances = pd.concat(df_all_occurances)





from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(df_all_cooccurances['speed'], df_all_cooccurances['pc1'],
                                              display_labels=['no plateau', 'plateau'], normalize='all')
plt.xlabel('Speed')
plt.ylabel('PC1')
plt.show()
















