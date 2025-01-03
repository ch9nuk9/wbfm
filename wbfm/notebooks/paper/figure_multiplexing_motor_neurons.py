#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[20]:


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
import plotly.express as px


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


# In[3]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP/ZIM2319_GFP_worm1-2022-12-10/project_config.yaml"
project_data_gfp = ProjectData.load_final_project_data_from_config(fname)


# In[5]:


# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_10-38_ZIM2165_immob_worm8-2022-12-13/project_config.yaml"
project_data_immob = ProjectData.load_final_project_data_from_config(fname)


# In[6]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
all_projects_gfp = load_paper_datasets('gfp')


# In[7]:



all_projects_immob = load_paper_datasets('immob')


# # Triple plots

# ## Example: WBFM

# In[8]:


from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter


# In[ ]:





# In[9]:


wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], ylim=[-0.33, 0.24])


# In[10]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm'
wbfm_plotter.plot_triple_traces('VB02', title=True, legend=True, output_foldername=output_foldername)
wbfm_plotter.plot_single_trace('VB02', title=False, legend=False, output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), width_factor=0.4)


# In[11]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm'
wbfm_plotter.plot_triple_traces('DB01', title=True, legend=True, output_foldername=output_foldername)
wbfm_plotter.plot_single_trace('DB01', title=False, legend=False, output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), width_factor=0.4)


# In[12]:


# For the supp
output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm'
wbfm_plotter.plot_triple_traces('BAGL', ylim=None, title=False, legend=True, 
                                output_foldername=output_foldername,width_factor=0.4, height_factor=0.2)
wbfm_plotter.plot_single_trace('BAGL', title=False, legend=False, output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), width_factor=0.4, height_factor=0.2)


# # Example: immob

# In[13]:


immob_plotter = PaperExampleTracePlotter(project_data_immob, xlim=[100, 300])


# In[14]:


project_data_immob


# In[15]:


immob_plotter.project.physical_unit_conversion.volumes_per_second


# In[16]:


# project_data_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)
# project_data_immob.calc_paper_traces()
# project_data_immob.calc_paper_traces_global()
# project_data_immob.calc_paper_traces_residual()


# In[17]:


neurons = ['VB02', 'DB01']

output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob'

for n in neurons:

    immob_plotter.plot_triple_traces(n, output_foldername=output_foldername)
    immob_plotter.plot_single_trace(n, color_type='immob', output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), width_factor=0.4)


# In[ ]:





# In[ ]:





# # Triggered averages

# ## Initial calculations

# In[21]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[ ]:


triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_gcamp)


# In[ ]:


# %debug


# In[ ]:


# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_ava
# # Run if the behavior for the immobilized isn't there or needs to be updated
# for p in all_projects_immob.values():
#     try:
#         beh_vec = approximate_behavioral_annotation_using_ava(p)
#     except:
#         print("Dataset failed")
#         print(p.shortened_name)


# In[ ]:


triggered_average_immob_plotter = PaperMultiDatasetTriggeredAverage(all_projects_immob)


# ## Motor

# In[ ]:


# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('VB02', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# In[ ]:


# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('DB01', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# In[ ]:


from wbfm.utils.general.utils_paper import apply_figure_settings
# Plot neurons as a rainbow on a single plot

# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

trigger_type = 'residual_rectified_fwd'
title = ''

# neurons = ['VB03', 'VB02', 'RMED', 'RMDVL', 'SMDDL', 'DB01', 'RMEV']
neurons = ['VB02', 'RMED', 'RMDV', 'SMDD', 'DB01', 'RMEV', 'VB03']

fig, ax = None, None
cmap = px.colors.qualitative.Plotly
for i, neuron in tqdm(enumerate(neurons)):
    color = cmap[i]
    fig, ax = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, fig=fig, ax=ax, use_plotly=True,
                                                                         title=title, show_title=True, ylim=[-0.09, 0.055], color=color,
                                                                         output_folder=None,#'/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                        i_figure=4)


# In[ ]:


apply_figure_settings(fig, plotly_not_matplotlib=True, width_factor=1.0, height_factor=0.3)
fig.update_xaxes(range=[-3, 3])
fig.update_yaxes(range=[-0.1, 0.1])
fig.update_layout(showlegend=True)
fig.show()


# ## BAG

# In[ ]:


# Actually plotted in the O2 multiplexing notebook

# trigger_types = [('global_rev', 'Reversal Triggered'), 
#                 ('residual_collision', 'Collision Triggered')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('BAGL', trigger_type, 
#                                                                          title=title, show_title=True,
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         fig_opt=dict(width_factor=0.5,height_factor=0.2))


# ## All O2 neurons

# In[ ]:


from wbfm.utils.general.hardcoded_paths import list_of_gas_sensing_neurons

trigger_types = [('global_rev', 'Reversal Triggered'), 
                ('residual_collision', 'Collision Triggered')]

for n in list_of_gas_sensing_neurons():
    print(n)
    for trigger_type, title in trigger_types:
        triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(n, trigger_type, 
                                                                             title="",#title,
                                                                             output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                            i_figure=5)


# In[ ]:





# # Plot of all neurons with signal (autocovariance)
# 

# In[ ]:


from wbfm.utils.visualization.multiproject_wrappers import plot_variance_all_neurons


# In[ ]:


output_folder = 'multiplexing'
fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
                                                                   loop_not_facet_row=True,
                                                                     names_to_keep_in_simple_id=('VB02', 'DB01'),
                                                                   use_paper_options=True, include_gfp=True, include_legend=True)


# In[ ]:


# output_folder = None
# fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
#                                                                    loop_not_facet_row=True,
#                                                                      names_to_keep_in_simple_id=('AVA', 'AVE', 'RIM', 'ALA'),
#                                                                    use_paper_options=True, include_gfp=True, include_legend=True)


# ## Total count of significant neurons in each data category

# In[ ]:


df_significant_numbers = df_summary.groupby('Type of data')['Significant'].value_counts().reset_index()
df_significant_numbers


# In[ ]:


df_total_numbers = df_significant_numbers.groupby('Type of data').sum().reset_index().drop(columns=['Significant'])
df_total_numbers


# In[ ]:


df_signficant_summary = df_significant_numbers.merge(df_total_numbers, on='Type of data')
df_signficant_summary


# In[ ]:


df_signficant_summary['percent_significant'] = df_signficant_summary['count_x'] / df_signficant_summary['count_y']
df_signficant_summary


# ## Count of significant neurons in residual AND global, and high (or low) PC correlation

# In[ ]:


df_subset = df_summary[(df_summary['Type of data']=='global gcamp') | (df_summary['Type of data']=='residual gcamp')]
df_subset = df_subset[df_subset['pc0_high'] | df_subset['pc0_low']]
df_subset = df_subset[df_subset['Significant']].reset_index()
df_subset.head()


# In[ ]:


# Count which indices are doubled, meaning both residual and global were significant
counts = df_subset.groupby('index')['Significant'].count()
result = counts[counts == 2].index
count_of_true_for_both_versions = len(result)
print(count_of_true_for_both_versions, count_of_true_for_both_versions/2682)


# In[ ]:


# df_subset.merge(df_total_numbers, on='Type of data')


# In[ ]:





# # Additional triple plots and triggered averages (examples)

# ## Additional triple plots (example dataset)

# In[ ]:


# wbfm_plotter.plot_triple_traces('RID', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('AVAL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('IL2LL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('IL2LR', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('RIS', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('RIVL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('RMED', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('RMEV', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('BAGR', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('VB01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


# wbfm_plotter.plot_triple_traces('VB03', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('DB02', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('VA02', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('VA01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# In[ ]:


wbfm_plotter.plot_triple_traces('DA01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# ## Additional triggered averages (multiple datasets)

# In[ ]:


trigger_types = [('global_rev', ''), 
                ('residual_rectified_rev', 'Reversal'),
                ('residual_rectified_fwd', 'Forward'),
                ('residual', '')]

for neuron in ['VB01', 'VB03', 'DB02', 'VA02', 'VA01', 'DA01','VB02']:
    for trigger_type, title in trigger_types:
        triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, 
                                                                             title=title, show_title=True, #ylim=[-0.09, 0.055],
                                                                             output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                            i_figure=4)
        print(neuron)
        plt.show()


# In[ ]:


# trigger_types = [('residual', '')]

# ax = None
# for neuron in ['RIVL']:
#     for trigger_type, title in trigger_types:
#         fig, ax = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                              title=title, show_title=True, #ylim=[-0.09, 0.055],
#                                                                              #output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                             i_figure=4, ax=ax)


# In[ ]:





# In[ ]:





# ## Alternate version of the cluster figure

# In[ ]:


trigger_types = [('raw_fwd', 'Forward Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    if 'raw' in trigger_type:
        xlim = (-20, 60)
    else:
        xlim = (-20, 20)
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('RIVL', trigger_type, 
                                                                         xlim=xlim,
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')


# In[ ]:


trigger_types = [('raw_fwd', 'Forward Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    if 'raw' in trigger_type:
        xlim = (-20, 60)
    else:
        xlim = (-20, 20)
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('RMED', trigger_type, 
                                                                         xlim=xlim,
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')


# In[ ]:


trigger_types = [('raw_fwd', 'Forward Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    if 'raw' in trigger_type:
        xlim = (-20, 60)
    else:
        xlim = (-20, 20)
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('RID', trigger_type,  
                                                                         xlim=xlim,
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')


# In[ ]:


trigger_types = [('raw_rev', 'Reversal Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    if 'raw' in trigger_type:
        xlim = (-20, 60)
    else:
        xlim = (-20, 20)
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('RIS', trigger_type, 
                                                                         xlim=xlim,
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')


# In[ ]:


trigger_types = [('raw_rev', 'Reversal Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('AVAL', trigger_type, 
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')


# In[ ]:





# In[ ]:





# # DEBUG

# In[ ]:


# y = immob_plotter.df_traces['BAGL']
# y2 = immob_plotter.df_traces_global['BAGL']
# y3 = immob_plotter.df_traces_residual['BAGL']

# df = pd.DataFrame({'y': y, 'resid': y3, 'global': y2})
# px.line(df)


# In[ ]:


beh_vec = immob_plotter.project.worm_posture_class.beh_annotation(fluorescence_fps=True)


# In[ ]:




