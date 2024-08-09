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
import plotly.express as px


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP/ZIM2319_GFP_worm1-2022-12-10/project_config.yaml"
project_data_gfp = ProjectData.load_final_project_data_from_config(fname)


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13/project_config.yaml"
project_data_immob = ProjectData.load_final_project_data_from_config(fname)


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
all_projects_gfp = load_paper_datasets('gfp')



all_projects_immob = load_paper_datasets('immob')


# # Triple plots

# ## Example: WBFM

from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter





wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], ylim=[-0.33, 0.24])


wbfm_plotter.plot_triple_traces('VB02', title=True, legend=True, output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('DB01', title=True, legend=True, output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('BAGL', ylim=None, title=True, legend=True, output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# # Example: immob

immob_plotter = PaperExampleTracePlotter(project_data_immob, xlim=[0, 200])


immob_plotter.project.physical_unit_conversion.volumes_per_second


# project_data_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)
# project_data_immob.calc_paper_traces()
# project_data_immob.calc_paper_traces_global()
# project_data_immob.calc_paper_traces_residual()


immob_plotter.plot_triple_traces('VB02', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob')


# immob_plotter.plot_triple_traces('RID', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob')


# immob_plotter.plot_triple_traces('AVAL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob')


# immob_plotter.plot_triple_traces('BAGL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob')


# immob_plotter.plot_triple_traces('DB01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/immob')








# # Triggered averages

# ## Initial calculations

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_gcamp)


# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_ava
# # Run if the behavior for the immobilized isn't there or needs to be updated
# for p in all_projects_immob.values():
#     try:
#         beh_vec = approximate_behavioral_annotation_using_ava(p)
#     except:
#         print("Dataset failed")
#         print(p.shortened_name)


triggered_average_immob_plotter = PaperMultiDatasetTriggeredAverage(all_projects_immob)


# ## Motor

# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('VB02', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('DB01', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# ## BAG

# trigger_types = [('global_rev', 'Reversal Triggered'), 
#                 ('residual_collision', 'Collision Triggered')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('BAGL', trigger_type, 
#                                                                          title="",#title,
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=5)





# # FFT of VB02 in immob and freely moving

from wbfm.utils.projects.finished_project_data import plot_frequencies_for_fm_and_immob_projects


output_folder = 'multiplexing'



opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'VB02', 
                                                                                                    output_folder=output_folder,**opt)



opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'AVAL', 
                                                                                                    output_folder=output_folder, **opt)


# # Plot of all neurons with signal (autocovariance)
# 

from wbfm.utils.visualization.multiproject_wrappers import plot_variance_all_neurons


output_folder = 'multiplexing'
fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
                                                                   loop_not_facet_row=True,
                                                                     names_to_keep_in_simple_id=('VB02', 'DB01'),
                                                                   use_paper_options=True, include_gfp=True, include_legend=True)


# output_folder = None
# fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
#                                                                    loop_not_facet_row=True,
#                                                                      names_to_keep_in_simple_id=('AVA', 'AVE', 'RIM', 'ALA'),
#                                                                    use_paper_options=True, include_gfp=True, include_legend=True)


# ## Total count of significant neurons in each data category

df_significant_numbers = df_summary.groupby('Type of data')['Significant'].value_counts().reset_index()
df_significant_numbers


df_total_numbers = df_significant_numbers.groupby('Type of data').sum().reset_index().drop(columns=['Significant'])
df_total_numbers


df_signficant_summary = df_significant_numbers.merge(df_total_numbers, on='Type of data')
df_signficant_summary


df_signficant_summary['percent_significant'] = df_signficant_summary['count_x'] / df_signficant_summary['count_y']
df_signficant_summary


# ## Count of significant neurons in residual AND global, and high (or low) PC correlation

df_subset = df_summary[(df_summary['Type of data']=='global gcamp') | (df_summary['Type of data']=='residual gcamp')]
df_subset = df_subset[df_subset['pc0_high'] | df_subset['pc0_low']]
df_subset = df_subset[df_subset['Significant']].reset_index()
df_subset.head()


# Count which indices are doubled, meaning both residual and global were significant
counts = df_subset.groupby('index')['Significant'].count()
result = counts[counts == 2].index
count_of_true_for_both_versions = len(result)
print(count_of_true_for_both_versions, count_of_true_for_both_versions/2682)


# df_subset.merge(df_total_numbers, on='Type of data')





# # Additional triple plots and triggered averages (examples)

# ## Additional triple plots (example dataset)

# wbfm_plotter.plot_triple_traces('RID', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('AVAL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('IL2LL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('IL2LR', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('RIS', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('RIVL', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('RMED', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('RMEV', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('BAGR', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('VB01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# wbfm_plotter.plot_triple_traces('VB03', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('DB02', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('VA02', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('VA01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


wbfm_plotter.plot_triple_traces('DA01', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# ## Additional triggered averages (multiple datasets)

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


# trigger_types = [('residual', '')]

# ax = None
# for neuron in ['RIVL']:
#     for trigger_type, title in trigger_types:
#         fig, ax = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                              title=title, show_title=True, #ylim=[-0.09, 0.055],
#                                                                              #output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                             i_figure=4, ax=ax)








# ## Alternate version of the cluster figure

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


trigger_types = [('raw_rev', 'Reversal Triggered'), 
                ('residual_rectified_rev', 'Reversal Rectified'),
                ('residual_rectified_fwd', 'Forward Rectified'),
                ('residual', 'Undulation Triggered')]

for trigger_type, title in trigger_types:
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('AVAL', trigger_type, 
                                                                         title=title,
                                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger')








# # DEBUG

# y = immob_plotter.df_traces['BAGL']
# y2 = immob_plotter.df_traces_global['BAGL']
# y3 = immob_plotter.df_traces_residual['BAGL']

# df = pd.DataFrame({'y': y, 'resid': y3, 'global': y2})
# px.line(df)


beh_vec = immob_plotter.project.worm_posture_class.beh_annotation(fluorescence_fps=True)




