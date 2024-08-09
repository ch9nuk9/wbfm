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


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_O2_fm = load_paper_datasets(['hannah_O2_fm', 'gcamp'])
all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')
all_projects_O2_fm_mutant = load_paper_datasets('hannah_O2_fm_mutant')
all_projects_O2_immob_mutant = load_paper_datasets('hannah_O2_immob_mutant')


# # Check ID'ed neurons

all_projects_ids = {}
for name, p in all_projects_O2_fm.items():
    ids = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0)
    if len(ids) > 4:
        all_projects_ids[name] = p
        print(name, len(ids))
    # else:
    #     print(name, ids)


# # Check that the behavior annotations make sense

from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior


# p = all_projects_ids['2023-11-21_14-46_wt_worm4-2023-11-21']
# p.plot_neuron_with_kymograph('neuron_010')

# speed = p.worm_posture_class.worm_speed(fluorescence_fps=True, strong_smoothing_before_derivative=True, signed=False)
# ang_speed = p.worm_posture_class.worm_angular_velocity(fluorescence_fps=True)

# # px.line(speed)
# # px.line(speed).show()

# plt.plot(10*speed)
# plt.plot(10*ang_speed)


# speed = p.worm_posture_class.worm_speed(fluorescence_fps=True, strong_smoothing_before_derivative=True, signed=False)
# df = pd.DataFrame({'speed': speed, 'speed+1': speed.shift(1), 'speed-1': speed.shift(-1)})
# df['nan'] = (df.abs() < 0.001).all(axis=1).astype(int)
# fig = px.line(df)
# fig.show()


# beh_vec = p.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True, use_pause_to_exclude_other_states=True)
# # beh_vec = pd.Series(p.worm_posture_class.tracking_failure_idx)
# beh_vec.iloc[1390:1400]





# for name, p in all_projects_O2_fm.items():
#     # print(p.worm_posture_class)
#     name_dict = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0, flip_names_and_ids=True)
#     neuron = name_dict.get('AVAL', None)
#     print(p.shortened_name, neuron)
#     plt.show()
#     if neuron is not None:
#         p.plot_neuron_with_kymograph(neuron)
#     # break


# for name, p in all_projects_O2_fm_mutant.items():
#     # print(p.worm_posture_class)
#     name_dict = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0, flip_names_and_ids=True)
#     neuron = name_dict.get('AVAL', None)
#     print(p.shortened_name, neuron)
#     plt.show()
#     if neuron is not None:
#         p.plot_neuron_with_kymograph(neuron)
#     # break


# # FM

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm, calculate_residual=False, calculate_global=False, calculate_turns=False, calculate_self_collision=True,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), 
                                                                    trigger_opt=dict(fixed_num_points_after_event=40)
                                                                   )


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e



# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('self_collision', 'Self-collision')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm',
#                                                                          min_lines=1, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # Immob

# all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


triggered_average_gcamp_plotter_immob = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
                                                                   calculate_stimulus=True, 
                                                                          trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6))


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e



# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# ## Same but downshift

triggered_average_gcamp_plotter_immob_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
                                                                   calculate_stimulus=True, 
                                                                                    trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6))


# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_downshift.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift',
#                                                                                      min_lines=1, xlim=[-5, 15],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_downshift.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # FM_mutant

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


triggered_average_gcamp_plotter_fm_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False, calculate_self_collision=True,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), 
                                                                    trigger_opt=dict(fixed_num_points_after_event=40)
                                                                   )


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm_mutant',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4, is_mutant=True)
#             except Exception as e:
#                 print(e)
#                 raise e


# # Plot all on one plot
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('self_collision', 'Self-collision')]
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm_mutant',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # immob_mutant

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# all_projects_O2_immob_mutant = load_paper_datasets('hannah_O2_immob_mutant')


triggered_average_gcamp_plotter_immob_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), calculate_stimulus=True, 
                                                                    trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6)
                                                                   )


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_mutant.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4, is_mutant=True)
#             except Exception as e:
#                 print(e)
#                 # raise e


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_mutant.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# ## Same but downshift

triggered_average_gcamp_plotter_immob_mutant_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
                                                                   calculate_stimulus=True, trigger_opt=dict(fixed_num_points_after_event=30, trigger_on_downshift=True, ind_delay=6))


# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_mutant_downshift.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant_downshift',
#                                                                                      min_lines=1, xlim=[-5, 15],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 # raise e


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_mutant_downshift.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant_downshift',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # Both WT and mutant on the same plot

# URX type neuron, i.e. responds to O2 upshift and forward behavior
neuron_list = ['AQR', 'URX','AUA', 'IL1L', 'IL2L']
is_mutant_vec = [False, True]

trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]
for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
                                                         title=f"O2 Upshift", show_title=True, #ylim=[-0.05, 0.5],
                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
                                                         min_lines=1, xlim=[-5, 12], ax=ax, 
                                                           round_y_ticks=True, show_y_ticks=True, show_y_label=False,
                                                         i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron))


trigger_type = 'raw_fwd'
plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]
for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
                                                         title=f"Forward Triggered", show_title=True, #ylim=[-0.05, 0.5],
                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
                                                         min_lines=1, xlim=[-5, 12], ax=ax, round_y_ticks=True, show_y_label=False,
                                                         i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron))


# Type 2: BAG-type, which responds to O2 downshift and reversals
neuron_list = ['BAG', 'ANTIcor', 'RMDV']
is_mutant_vec = [False, True]

trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_downshift, triggered_average_gcamp_plotter_immob_mutant_downshift]

for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
                                                         title=f"O2 Downshift", show_title=True, #ylim=[-0.05, 0.5],
                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
                                                         min_lines=1, xlim=[-5, 12], ax=ax, 
                                                           round_y_ticks=True, show_y_ticks=True, show_y_label=False,
                                                         i_figure=4, is_mutant=is_mutant, show_x_ticks=False)

trigger_type = 'raw_rev'
plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]

for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
                                                         title=f"Reversal Triggered", show_title=True, #ylim=[-0.05, 0.5],
                                                         output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
                                                         min_lines=1, xlim=[-5, 12], ax=ax, round_y_ticks=True, show_y_label=False,
                                                         i_figure=4, is_mutant=is_mutant, show_x_ticks=False)


# ## Just export legend

from wbfm.utils.general.utils_paper import export_legend_for_paper

export_legend_for_paper('/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/legend.png')


# ## Same list of neurons with example time series

from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter


project_data_gcamp = all_projects_O2_fm['ZIM2165_Gcamp7b_worm1-2022_11_28']
# For ANTIcorL and RMDV
project_data_gcamp2 = all_projects_O2_fm['2023-11-30_14-31_wt_worm5_FM-2023-11-30']


wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120])#, ylim=[-0.35, 0.35])
wbfm_plotter2 = PaperExampleTracePlotter(project_data_gcamp2, xlim=[50, 170])#, ylim=[-0.35, 0.35])


neuron_list = ['AQR', #'AUAL', 
               'IL1LR', 'IL2LL', #'RMDVL', 
               'BAGL',
              'URXL', ]#'ANTIcorR']

output_folder = os.path.join('multiplexing', 'o2_example_traces')

for i, neuron in enumerate(neuron_list):
    if 'URX' in neuron:
        xlabels=True
    else:
        xlabels=False
    try:
        print(neuron)
        wbfm_plotter.plot_single_trace(neuron, title=False, round_y_ticks=True, xlabels=xlabels,
                                       output_foldername=output_folder)
        plt.show()
    except ValueError as e:
        # print(e)
        pass


# Second set of neurons
neuron_list = ['AUAL', 'AUAR', 'RMDVL', 'RMDVR',
               'ANTIcorR', 'ANTIcorL']

output_folder = os.path.join('multiplexing', 'o2_example_traces')

for i, neuron in enumerate(neuron_list):
    if 'URX' in neuron:
        xlabels=True
    else:
        xlabels=False
    try:
        print(neuron)
        wbfm_plotter2.plot_single_trace(neuron, title=False, round_y_ticks=True, xlabels=xlabels,
                                       output_foldername=output_folder)
        plt.show()
    except ValueError as e:
        # print(e)
        pass


# ## Do ttests to compare the triggered averages

from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
from wbfm.utils.general.utils_paper import apply_figure_settings


# Type 2: BAG-type, which responds to O2 downshift and reversals
neuron_list = ['BAG', 'ANTIcor', 'RMDV']
is_mutant_vec = [False, True]

all_boxplot_data_dfs = []

trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_downshift, triggered_average_gcamp_plotter_immob_mutant_downshift]

# for neuron in neuron_list:
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         means_before, means_after = obj.get_boxplot_before_and_after(neuron, trigger_type)
        
#         df_before = pd.DataFrame(means_before, columns=['mean']).assign(before=True)
#         df_after = pd.DataFrame(means_after, columns=['mean']).assign(before=False)
#         df_both = pd.concat([df_before, df_after]).assign(neuron=neuron, is_mutant=is_mutant)
#         all_boxplot_data_dfs.append(df_both)

trigger_type = 'raw_rev'
plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]

for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        means_before, means_after = obj.get_boxplot_before_and_after(neuron, trigger_type, same_size_window=False)
        
        df_before = pd.DataFrame(means_before, columns=['mean']).assign(before=True)
        df_after = pd.DataFrame(means_after, columns=['mean']).assign(before=False)
        df_both = pd.concat([df_before, df_after]).assign(neuron=neuron, is_mutant=is_mutant).assign(trigger_type='rev')
        all_boxplot_data_dfs.append(df_both)

df_boxplot = pd.concat(all_boxplot_data_dfs)


# Type 2: BAG-type, which responds to O2 downshift and reversals
neuron_list = ['AQR', 'URX', 'AUA', 'IL1L', 'IL2L']
is_mutant_vec = [False, True]

# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]

# all_boxplot_data_dfs = []

# for neuron in neuron_list:
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         means_before, means_after = obj.get_boxplot_before_and_after(neuron, trigger_type)
        
#         df_before = pd.DataFrame(means_before, columns=['mean']).assign(before=True)
#         df_after = pd.DataFrame(means_after, columns=['mean']).assign(before=False)
#         df_both = pd.concat([df_before, df_after]).assign(neuron=neuron, is_mutant=is_mutant)
#         all_boxplot_data_dfs.append(df_both)

trigger_type = 'raw_fwd'
plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]

for neuron in neuron_list:
    ax = None
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        means_before, means_after = obj.get_boxplot_before_and_after(neuron, trigger_type, same_size_window=False)
        
        df_before = pd.DataFrame(means_before, columns=['mean']).assign(before=True)
        df_after = pd.DataFrame(means_after, columns=['mean']).assign(before=False)
        df_both = pd.concat([df_before, df_after]).assign(neuron=neuron, is_mutant=is_mutant).assign(trigger_type='fwd')
        all_boxplot_data_dfs.append(df_both)

df_boxplot = pd.concat(all_boxplot_data_dfs)


# df = df_boxplot[~df_boxplot['is_mutant']]

# fig = px.box(df, x='neuron', y='mean', color='before', points='all')
# add_p_value_annotation(fig, x_label='all', show_ns=True, permutations=1000)#, _format=dict(text_height=0.075))
# apply_figure_settings(fig, height_factor=0.2)
# fig.show()


# df = df_boxplot[df_boxplot['is_mutant']]

# fig = px.box(df, x='neuron', y='mean', color='before', points='all')
# add_p_value_annotation(fig, x_label='all', show_ns=True, permutations=1000, DEBUG=False)#, _format=dict(text_height=0.075))
# apply_figure_settings(fig, height_factor=0.2)
# fig.show()


# fig.data


# ## Same boxplots, but per neuron and correct colors

from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, plot_box_multi_axis


# Make a new column with color information based on reversal
def add_columns_to_df_for_boxplots(neuron_name, is_rev_triggered=True):
    df = df_boxplot[df_boxplot['neuron']==neuron_name].copy()
    
    if is_rev_triggered:
        before_str, after_str = 'Fwd', 'Rev'
    else:
        before_str, after_str = 'Rev', 'Fwd'
    
    df['color'] = ''
    df.loc[np.logical_and(df['before'], df['is_mutant']), 'color'] = f'{before_str}-Mutant'
    df.loc[np.logical_and(~df['before'], df['is_mutant']), 'color'] = f'{after_str}-Mutant'
    df.loc[np.logical_and(df['before'], ~df['is_mutant']), 'color'] = f'{before_str}-WT'
    df.loc[np.logical_and(~df['before'], ~df['is_mutant']), 'color'] = f'{after_str}-WT'
    df['is_mutant_str'] = 'gcy-31; -35; -9'
    df.loc[~df['is_mutant'], 'is_mutant_str'] = 'Wild Type'

    # Rename columns to the display names
    df['Data Type'] = df['color']
    df['dR/R50'] = df['mean']
    
    if is_rev_triggered:
        df['before_str'] = ['FWD' if val else 'REV' for val in df['before']]
    else:
        df['before_str'] = ['REV' if val else 'FWD' for val in df['before']]

    return df
    



cmap = plotly_paper_color_discrete_map()

for neuron_name in ['BAG', 'ANTIcor', 'RMDV']:

    df = add_columns_to_df_for_boxplots(neuron_name, is_rev_triggered=True)

    fig = plot_box_multi_axis(df, x_columns_list=['is_mutant_str', 'before_str'], y_column='mean',
                             color_names=['Wild Type', 'gcy-31; -35; -9'], DEBUG=False)
    add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=False, bonferroni_factor=8,
                           height_mode='top_of_data', has_multicategory_index=True, DEBUG=False)

    # fig = px.box(df, x='is_mutant_str', y='dR/R50', #color='before',
    #                   color='Data Type', color_discrete_map={'Rev-Mutant':cmap['mutant'], 'Rev-WT':cmap['gcamp'], 
    #                                                      'Fwd-Mutant':cmap['mutant'], 'Fwd-WT':cmap['gcamp']}, hover_data=['before'],)
    # add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=True, bonferroni_factor=8,
    #                        height_mode='top_of_data', DEBUG=False)#, _format=dict(text_height=0.075))
    fig.update_xaxes(title='')
    fig.update_yaxes(title=r'$\Delta R / R_{50}$')  # Already exists for the triggered averages themselves
    fig.update_layout(showlegend=False)
    # Modify offsetgroup to have only 2 types (rev and fwd), not one for each legend entry
    # for d in fig.data:
    #     d['offsetgroup'] = 'Fwd' if 'Fwd' in d['offsetgroup'] else 'Rev'
    apply_figure_settings(fig, height_factor=0.15, width_factor=0.35)
    fig.show()

    output_dir = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant"
    fname = os.path.join(output_dir, f'{neuron_name}_triggered_average_boxplots.png')
    fig.write_image(fname)#, scale=4)
    fname = fname.replace('.png', '.svg')
    fig.write_image(fname)




for neuron_name in ['AQR', 'URX','AUA', 'IL1L', 'IL2L']:

    df = add_columns_to_df_for_boxplots(neuron_name, is_rev_triggered=False)

    fig = plot_box_multi_axis(df, x_columns_list=['is_mutant_str', 'before_str'], y_column='mean',
                             color_names=['Wild Type', 'gcy-31; -35; -9'], DEBUG=False)
    add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=False, bonferroni_factor=8,
                           height_mode='top_of_data', has_multicategory_index=True, DEBUG=False)
    # fig = px.box(df, x='is_mutant_str', y='dR/R50', #color='before',
    #                   color='Data Type', color_discrete_map={'Rev-Mutant':cmap['mutant'], 'Rev-WT':cmap['gcamp'], 
    #                                                      'Fwd-Mutant':cmap['mutant'], 'Fwd-WT':cmap['gcamp']}, hover_data=['before'],)
    # add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=True, bonferroni_factor=8,
    #                        height_mode='top_of_data', DEBUG=False)#, _format=dict(text_height=0.075))
    fig.update_xaxes(title='')
    fig.update_yaxes(title=r'$\Delta R / R_{50}$')  # Already exists for the triggered averages themselves
    fig.update_layout(showlegend=False)
    # Modify offsetgroup to have only 2 types (rev and fwd), not one for each legend entry
    # for d in fig.data:
    #     d['offsetgroup'] = 'Fwd' if 'Fwd' in d['offsetgroup'] else 'Rev'
    apply_figure_settings(fig, height_factor=0.15, width_factor=0.35)
    fig.show()

    output_dir = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant"
    fname = os.path.join(output_dir, f'{neuron_name}_triggered_average_boxplots.png')
    fig.write_image(fname)#, scale=4)
    fname = fname.replace('.png', '.svg')
    fig.write_image(fname)


# fig = plot_box_multi_axis(df, x_columns_list=['is_mutant_str', 'before_str'], y_column='mean')
# add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=False, bonferroni_factor=8,
#                            height_mode='top_of_data', has_multicategory_index=True, DEBUG=False)
# apply_figure_settings(fig, width_factor=0.3, height_factor=0.3)
# fig.update_layout(showlegend=False)
# fig.show()


# list(fig.data[0].x)








# ## Debug

# ## Debugging: triggered average on single dataset

# p = all_projects_O2_immob['2023-09-19_11-42_worm1-2023-09-19']


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages


p = all_projects_O2_immob['2023-09-07_16-11_CaMP7b_O2_worm1-2023-09-07']
p.use_physical_time = True

triggered_avg_class = FullDatasetTriggeredAverages.load_from_project(p, trace_opt=dict(use_paper_options=False, rename_neurons_using_manual_ids=True, high_pass_bleach_correct=False, manual_id_confidence_threshold=0), 
                                                                     trigger_opt=dict(state=BehaviorCodes.STIMULUS, trigger_on_downshift=True, ind_delay=0))


triggered_avg_class.plot_events_over_trace('AQR')


triggered_avg_class.plot_events_over_trace('URXR')


triggered_avg_delay_class = FullDatasetTriggeredAverages.load_from_project(p, trace_opt=dict(use_paper_options=False, rename_neurons_using_manual_ids=True, high_pass_bleach_correct=False, manual_id_confidence_threshold=0), 
                                                                     trigger_opt=dict(state=BehaviorCodes.STIMULUS, trigger_on_downshift=True, ind_delay=6))


triggered_avg_delay_class.plot_events_over_trace('AQR')


ax = triggered_avg_delay_class.plot_single_neuron_triggered_average('BAG')
triggered_avg_class.plot_single_neuron_triggered_average('BAG', ax=ax)


# triggered_avg_delay_class.plot_single_neuron_triggered_average('BAG', DEBUG=True)
triggered_avg_delay_class.triggered_average_matrix_from_name('BAGL', DEBUG=True)


triggered_avg_class.triggered_average_matrix_from_name('BAGL', DEBUG=True)


# triggered_avg_class.ind_class.to_nan_points_of_state_before_point, triggered_avg_delay_class.ind_class.to_nan_points_of_state_before_point


triggered_avg_delay_class.ind_class._get_invalid_states_for_prior_index_removal()


triggered_avg_class.ind_class._get_invalid_states_for_prior_index_removal()


triggered_avg_class.plot_single_neuron_triggered_average('BAG', DEBUG=True)








triggered_avg_delay_class


triggered_avg_class


df_stim = triggered_avg_delay_class.ind_class.cleaned_binary_state.reset_index(drop=True).astype(int)
df_stim.index = triggered_avg_class.df_traces.index
df = pd.concat([df_stim, triggered_avg_class.df_traces], axis=1)

fig = px.line(df, y=['RMDVR', 'URXR', 'BAGL', 'AQR', 'PQR', 'ANTIcorL', 0])
fig.show()

# fname = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob'


triggered_avg_class.ind_class.idx_onsets, triggered_avg_delay_class.ind_class.idx_onsets


y = triggered_avg_class.df_traces['URXR']
list(y.index)[500]


all_projects_O2_immob['2023-09-07_16-11_CaMP7b_O2_worm1-2023-09-07']


p.physical_unit_conversion




