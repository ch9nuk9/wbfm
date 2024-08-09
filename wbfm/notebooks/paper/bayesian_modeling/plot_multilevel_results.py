#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os 
import numpy as np
from pathlib import Path
import plotly.express as px


from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
print(fname)
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
print(fname)
Xy_gfp = pd.read_hdf(fname)


'VG_post_turning_R' in Xy_gfp


Xy_gfp.head()


# # Plot model comparison statistics

# Load data from many dataframes
output_dir = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output'
# output_dir = os.path.join(get_hierarchical_modeling_dir(), 'output')
all_dfs = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        all_dfs[neuron_name] = pd.read_hdf(filename)
df = pd.concat(all_dfs).reset_index(names=['neuron_name', 'model_type'])


# Also load gfp
output_dir = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/output'

# output_dir = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'output')
all_dfs_gfp = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        all_dfs_gfp[neuron_name] = pd.read_hdf(filename)
df_gfp = pd.concat(all_dfs_gfp).reset_index(names=['neuron_name', 'model_type'])


from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
df_num_datasets = Xy.groupby('dataset_name').apply(lambda x: x.notnull().any()).sum().to_frame()
df_num_datasets['datatype'] = 'gcamp'
df_num_datasets_gfp = Xy_gfp.groupby('dataset_name').apply(lambda x: x.notnull().any()).sum().to_frame()
df_num_datasets_gfp['datatype'] = 'gfp'
df_num_datasets = pd.concat([df_num_datasets, df_num_datasets_gfp])
df_num_datasets.rename(columns={0: 'number'}, inplace=True)

px.bar(df_num_datasets.loc[neurons_with_confident_ids(), :].sort_values(by='number'), y='number', 
       color='datatype', barmode='group')


# from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
# df_num_datasets = Xy_gfp.groupby('dataset_name').apply(lambda x: x.notnull().any()).sum()
# px.scatter(df_num_datasets[c for c in neurons_with_confident_ids() if c in df_num_datasets.index].sort_values())


# df_pivot = df.pivot(columns='model_type', index='neuron_name', values='elpd_loo')
# df_pivot = df_pivot.divide(Xy.count(), axis=0).dropna()
# px.scatter(df_pivot, x='null', y='hierarchical_pca', text=df_pivot.index)


# What I want to plot:
# x = scaled difference between the null and non-hierarchical model
# y = same but for hierarchical
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping, package_bayesian_df_for_plot
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids

# df_to_plot_gcamp = package_bayesian_df_for_plot(df, relative_improvement=False).assign(datatype='Freely Moving (GCaMP, residual)')
# df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, relative_improvement=False).assign(datatype='Freely Moving (GFP, residual)')
df_to_plot_gcamp = package_bayesian_df_for_plot(df, df_normalization=Xy, relative_improvement=False,
                                                min_num_datapoints=10000).assign(datatype='Freely Moving (GCaMP, residual)')
df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, df_normalization=Xy_gfp, relative_improvement=False,
                                              min_num_datapoints=5000).assign(datatype='Freely Moving (GFP, residual)')

# Add a couple names back in
# df_to_plot_gfp.loc['VB02', 'text'] = 'VB02 (gfp)'
# df_to_plot_gfp.loc['RMED', 'text'] = 'RMED (gfp)'
# df_to_plot_gfp.loc['RMEV', 'text'] = 'RMEV (gfp)'
rename_func = lambda x: f'{x} (gfp)' if x != '' else ''
df_to_plot_gfp.loc[:, 'text'] = df_to_plot_gfp.loc[:, 'text'].apply(rename_func)
# df_to_plot_gcamp.loc['RMDVL', 'text'] = 'RMDVL'
# df_to_plot_gcamp.loc['SMDVR', 'text'] = 'SMDVR'
df_to_plot_gcamp.loc['VB03', 'text'] = 'VB03'
# Remove a couple names
# df_to_plot_gcamp.loc['BAGL', 'text'] = ''
df_to_plot_gcamp.loc['URADL', 'text'] = ''

df_to_plot = pd.concat([df_to_plot_gcamp, df_to_plot_gfp])
df_to_plot['Dataset Type'] = df_to_plot['datatype']
df_to_plot['Size'] = 1


# %debug


# y, x = 'Hierarchy Score', 'null_normalized'
y, x = 'Hierarchy Score', 'Behavior Score'
# y, x = 'Hierarchy Score', 'hierarchical_pca_normalized'

x_max_gfp = df_to_plot_gfp[x].max()
y_max_gfp = df_to_plot_gfp[y].max()
print('GFP thresholds: ', y_max_gfp, x_max_gfp)

def categorize_row(row):
    if row[y] > y_max_gfp and row[x] > x_max_gfp:
        return 'Hierarchical Behavior'
    elif row[y] <= y_max_gfp and row[x] > x_max_gfp:
        return 'Behavior only'
    elif row[y] > y_max_gfp and row[x] <= x_max_gfp:
        return 'Hierarchy only'
    else:
        return 'No Behavior or Hierarchy'

# Apply function to create new column
df_to_plot_gcamp['Category'] = df_to_plot_gcamp.apply(categorize_row, axis=1)
df_to_plot['Category'] = df_to_plot.apply(categorize_row, axis=1)

fig = px.scatter(df_to_plot, 
                 # x='Hierarchy Score', y='Behavior Score', range_y=[-2, 60],
                 y=y, x=x, #range_x=[-2, 60],
                 text='text', 
                 color='Category', #color='Dataset Type',
                color_discrete_map=plotly_paper_color_discrete_map(), size='Size', size_max=10,
                )
fig.update_traces(textposition='middle right')

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

# gfp lines
fig.add_shape(type="line",
              x0=x_max_gfp, y0=0,  # start of the line (bottom of the plot)
              x1=x_max_gfp, y1=1,  # end of the line (top of the plot)
              line=dict(color="black", width=2, dash="dash"),
              xref='x',
              yref='paper')
fig.add_shape(type="line",
              x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
              x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
              line=dict(color="black", width=2, dash="dash"),
              xref='paper',
              yref='y')
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.3
))
fig.update_xaxes(title=f'{x}')# over Behavior model')
fig.update_yaxes(title=f'{y}')# <br>over Trivial model')

to_save = False
if to_save:
    ##
    # Make a figure for presentations with fewer names
    ##
    apply_figure_settings(fig, height_factor=0.3, width_factor=0.5)
    # fig.show()  # Showing here messes it up for the next save
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_with_gfp_presentation.png')
    fig.write_image(fname, scale=7)


##
# Final settings
##

# # Add some additional annotations with arrows and offsets (gfp)
# annotations_to_add = ['VB02', 'RMED', 'RMEV']
# offset_list = [[10, -150], [250, -50], [130, -50]]
# for offset, neuron in zip(offset_list, annotations_to_add):
#     ind = df_to_plot['datatype'] == 'Freely Moving (GFP, residual)'
#     xy = list(df_to_plot[ind].loc[neuron, [x, y]])
#     text = f'{neuron} (GFP)'
#     fig.add_annotation(x=xy[0], y=xy[1], ax=offset[0], ay=offset[1],
#                        text=text, showarrow=True)
    
# # Add some additional annotations with arrows and offsets (gcamp)
# annotations_to_add = ['RIS']
# offset_list = [[100, -50]]
# for offset, neuron in zip(offset_list, annotations_to_add):
#     ind = df_to_plot['datatype'] == 'Freely Moving (GCaMP, residual)'
#     xy = list(df_to_plot[ind].loc[neuron, [x, y]])
#     text = f'{neuron}'
#     fig.add_annotation(x=xy[0], y=xy[1], ax=offset[0], ay=offset[1],
#                        text=text, showarrow=True)
    
apply_figure_settings(fig, height_factor=0.3, width_factor=1.0)

fig.show()

to_save = False
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_with_gfp.png')
    # fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH", 'hierarchy_behavior_score_with_gfp.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)



fig = px.pie(df_to_plot_gcamp, names='Category', color='Category',
         color_discrete_map=plotly_paper_color_discrete_map(),
           )
apply_figure_settings(fig, height_factor=0.2, width_factor=0.5)

fig.show()

to_save = False
if to_save:
   # output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH"
   output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots"
   fname = os.path.join(output_folder, 'hierarchy_explained_pie_chart.png')
   fig.write_image(fname, scale=4)
   fname = fname.replace('.png', '.svg')
   fig.write_image(fname)


# # Alternate axes: manifold variance



def calc_var_ratio(Xy):
    Xy_var = Xy.groupby('dataset_name').var()
    Xy_var_ratio = {}
    for col in Xy_var.columns:
        if 'neuron' in col:
            continue
        col_manifold = f'{col}_manifold'
        if col_manifold in Xy_var.columns:
            Xy_var_ratio[col] = Xy_var[col_manifold] / Xy_var[col]
    Xy_var_ratio = pd.DataFrame(Xy_var_ratio)
    return Xy_var_ratio
    
df_var_exp_gcamp = calc_var_ratio(Xy)
df_var_exp_gcamp['Dataset Type'] = 'Freely Moving (GCaMP, residual)'
df_var_exp_gfp = calc_var_ratio(Xy_gfp)
df_var_exp_gfp['Dataset Type'] = 'Freely Moving (GFP, residual)'
df_var_exp = pd.concat([df_var_exp_gcamp, df_var_exp_gfp], axis=0)
    
# px.box(df_var_exp.dropna(thresh=3, axis=1), color='Dataset Type')



df_var_exp_median = df_var_exp.groupby('Dataset Type').median().reset_index().melt(
    id_vars='Dataset Type', var_name='neuron_name', value_name='manifold_variance')
df_to_plot_with_var = df_to_plot.merge(df_var_exp_median, on=['neuron_name', 'Dataset Type'])

# df_to_plot_with_var


x, y = 'manifold_variance', 'Hierarchy Score'

# Define threshold(s)
# x_max_gfp = df_to_plot_gfp[x].max()
y_max_gfp = df_to_plot_gfp[y].max()
print('GFP thresholds: ', y_max_gfp)
df_to_plot_with_var['above_gfp'] = df_to_plot_with_var[y] > y_max_gfp
df_to_plot_with_var['text_simple'] = df_to_plot_with_var['text']
df_to_plot_with_var.loc[~df_to_plot_with_var['above_gfp'], 'text_simple'] = ''

# Actual plot
fig = px.scatter(df_to_plot_with_var, 
                 y=y, x=x,
                 text='text_simple', 
                 color='Dataset Type',
                color_discrete_map=plotly_paper_color_discrete_map(), size='Size', size_max=10,
                )
fig.add_shape(type="line",
              x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
              x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
              line=dict(color="black", width=2, dash="dash"),
              xref='paper',
              yref='y')
fig.update_xaxes(title='Variance Explained by Manifold')
fig.update_yaxes(title='Hierarchical Model Performance')
apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.5
))

fig.show()


to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_and_manifold_variance.png')
    # fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH", 'hierarchy_behavior_score_with_gfp.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# df_to_plot_with_var.head()


# # Additional subplots: model parameters

import arviz as az
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids

def load_all_traces(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except:
                pass
    return all_traces

parent_folder = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_eigenworms34_speed'
suffix = ''
            
foldername = os.path.join(parent_folder, f'output{suffix}')
all_traces_gcamp = load_all_traces(foldername)

foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
all_traces_gfp = load_all_traces(foldername)


# Plot all that make it above the gfp line
# neurons_to_plot = df_to_plot_with_var['neuron_name'][df_to_plot_with_var['Category'] != 'No Behavior or Hierarchy']
neurons_to_plot = neurons_with_confident_ids()
# neurons_to_plot = ['URXR', 'URXL']

# neurons_to_plot = ['SMDDL', 'SMDDR', 'VB02', 'DB01', 'DB02']
# neurons_to_plot = ['BAGL', 'BAGR', 'AVAL', 'AVAR']
# neurons_to_plot = fnames
# neurons_to_plot = ['SMDDL', 'SMDDR', 'VG_post_turning_R', 'VG_post_turning_L']
neurons_to_plot = list(set(neurons_to_plot).intersection(set(all_traces_gcamp.keys())))


# all_traces_gfp['URXL']


# var_names = ["self_collision", 'speed', 'eigenworm3', 'eigenworm4', 'amplitude_mu']
var_names = ["self_collision", 'dorsal', 'ventral', 'amplitude_mu']

all_traces = all_traces_gcamp
# all_traces = all_traces_gfp

# az.plot_forest([all_traces[n] for n in neurons_to_plot], model_names=neurons_to_plot,
#                var_names=var_names, combined=True, 
#               filter_vars='like', kind='ridgeplot', figsize=(9, 7), ridgeplot_overlap=3)


# Scatter plot of median model parameters
from collections import defaultdict
var_names = ["self_collision", 'amplitude_mu', 'eigenworm', 'speed', 'phase', 'dorsal', 'ventral']

all_dfs = {}
for n in tqdm(neurons_to_plot):
    dat = az.extract(all_traces[n], group='posterior', var_names=var_names, filter_vars='like')
    all_dfs[n] = dat.to_dataframe().drop(columns=['chain', 'draw']).median()


df_params = pd.concat(all_dfs, axis=1).T
df_params['dataset_type'] = 'residual'
df_params.head()


r = np.exp(df_params['log_amplitude_mu'])
text = np.array(df_params.index)
text[r < 0.2] = ''

fig = px.scatter_polar(df_params, r=r, theta='phase_shift', text=text,
                      color='dataset_type', color_discrete_map=plotly_paper_color_discrete_map())
fig.update_traces(thetaunit='radians', textposition='bottom center')

apply_figure_settings(fig, width_factor=0.4, height_factor=0.3)
fig.update_layout(polar=dict(
    angularaxis = dict(thetaunit = "radians"),
    radialaxis = dict(#title='Oscillation<br>Amplitude', 
                      nticks=3)
), 
                  showlegend=False, title='Oscillation Amplitude and Phase', margin=dict(t=40))

fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'phase_shift_and_oscillation_amplitude.png')
    # fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH", 'hierarchy_behavior_score_with_gfp.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


fig = px.scatter(df_params.sort_values(by='self_collision_coefficient'), y='self_collision_coefficient', #x=df_params.index,
                )#text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


fig = px.scatter(df_params, x='dorsal_only_head_curvature_coefficient', y='ventral_only_head_curvature_coefficient',
          text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


fig = px.scatter(df_params, x='dorsal_only_body_curvature_coefficient', y='ventral_only_body_curvature_coefficient',
          text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


fig = px.scatter(df_params, x='eigenworm3_coefficient', y='eigenworm4_coefficient',
          text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


# az.plot_density([all_traces_gcamp[n] for n in neurons_to_plot], data_labels=neurons_to_plot ,
#                var_names=var_names,
#               filter_vars='like', figsize=(15, 7))








# ## Check the variables used

import arviz as az

foldername = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output/'
fnames = ['URXL']
for neuron in tqdm(fnames):
    trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
    if os.path.exists(trace_fname):
        test_trace = az.from_netcdf(trace_fname)
            


test_trace


all_traces_gfp['URXL']


# # Model explanation (simplified cartoon)

from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir
from wbfm.utils.projects.finished_project_data import ProjectData


fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
Xy = pd.read_hdf(fname)


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28"
project_data = ProjectData.load_final_project_data_from_config(fname, verbose=0)

dataset_name = "ZIM2165_Gcamp7b_worm1-2022_11_28"


# dataset_name = Xy_ind_range.index[i_dataset]
idx = Xy['dataset_name'] == dataset_name
project_data.use_physical_time = True
x_range = [0, 120]


df_to_plot = Xy.loc[idx, :].reset_index(drop=True)
df_to_plot.index = project_data.x_for_plots#[:-1]


df_to_plot.head()


# SECOND STYLE: two plots on one 

def _set_options(fig, height_factor=0.1):
    fig.update_yaxes(title_text='z-score')#title_text=f'{beh}')
    fig.update_xaxes(title_text='Time (seconds)', range=x_range)
    fig.update_layout(showlegend=False)
    apply_figure_settings(fig, height_factor=height_factor, width_factor=0.3)
    project_data.shade_axis_using_behavior(plotly_fig=fig)
    fig.show()


# First, behavior
for i, beh in enumerate([['eigenworm0', 'eigenworm1'], ['eigenworm2', 'eigenworm3']]):
    fig = px.line(df_to_plot[beh], color_discrete_sequence=px.colors.qualitative.Set1)
    _set_options(fig)
    
    # fig.write_image(f'{beh}.png', scale=7)

# Second, pca modes
for beh in [['pca_0', 'pca_1']]:
    fig = px.line(df_to_plot[beh], color_discrete_sequence=px.colors.qualitative.Dark2)
    _set_options(fig, height_factor=0.2)
    
    # fig.write_image(f'{beh}.png', scale=7)

# Final, observed data
for y_name in ['VB02']:
    fig = px.line(df_to_plot[y_name], color_discrete_sequence=px.colors.qualitative.Dark2)
    _set_options(fig, height_factor=0.2)
    
    # fig.write_image(f'{y_name}-raw.png', scale=7)








# # Debug scores

_df = df_gfp

# Build properly index dfs for each
df_loo = _df.pivot(columns='model_type', index='neuron_name', values='elpd_loo')
df_se = _df.pivot(columns='model_type', index='neuron_name', values='se')
df_loo_scaled = df_loo / df_se

x = (df_loo_scaled['hierarchical_pca'] - df_loo_scaled['nonhierarchical']).clip(lower=0)
y = (df_loo_scaled['nonhierarchical'] - df_loo_scaled['null']).clip(lower=0)
text_labels = pd.Series(list(x.index), index=x.index)
no_label_idx = np.logical_and(x < 5, y < 8)  # Displays some blue-only text
# no_label_idx = y < 8
# text_labels[no_label_idx] = ''

df_to_plot = pd.DataFrame({'Hierarchy Score': x, 'Behavior Score': y, 'text': text_labels, 'neuron_name': x.index})


df.head()


df_weight = df.pivot(columns='model_type', index='neuron_name', values='elpd_diff').copy()#.reset_index()
df_weight = df_weight / df.pivot(columns='model_type', index='neuron_name', values='dse') 
df_weight['datatype'] = 'gcamp'
df_weight2 = df_gfp.pivot(columns='model_type', index='neuron_name', values='elpd_diff').copy()#.reset_index()
df_weight2 = df_weight2 / df_gfp.pivot(columns='model_type', index='neuron_name', values='dse') 
df_weight2['datatype'] = 'gfp'
df_weight = pd.concat([df_weight, df_weight2])
# df_weight['


px.scatter(df_weight, x='nonhierarchical', y='null', color='datatype', text=df_weight.index)


# df[df['model_type'] == 'hierarchical_pca']


df[df['neuron_name'] == 'AVAL']


df_gfp[df_gfp['neuron_name'] == 'AVAL']


df




