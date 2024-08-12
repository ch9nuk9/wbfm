#!/usr/bin/env python
# coding: utf-8

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
import plotly.express as px
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
import napari
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd


fname = '/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/projects/19062024/2024-06-19_16-08_mec4_7b_higher_red_worm4_tap-2024-06-19/behavior/raw_stack_AVG_background_subtracted_normalisedDLC_resnet50_wbfm_nose_tailJan4shuffle1_1030000.h5'
df = pd.read_hdf(fname)


# fname = "/home/charles/Current_work/collaboration/konstantinos/adult_1030.nd2"
# import nd2
# img = nd2.imread(fname)
# v = napari.view_image(img)


img.shape





# all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')
# 





fname = "/lisc/scratch/neurobiology/zimmer/wbfm/test_projects/freely_moving/pytest-raw/project_config.yaml"

p = ProjectData.load_final_project_data_from_config(fname)

p.project_config.get_folders_for_behavior_pipeline()


# fname = "/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/projects/13062024/2024-06-13_17-42_control_worm10_wflyback-2024-06-13/project_config.yaml"
fname = "/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/project_config.yaml"
p2 = ProjectData.load_final_project_data_from_config(fname)

# p = all_projects_O2_immob['2023-09-07_16-11_CaMP7b_O2_worm1-2023-09-07']
# p.use_physical_time = True


p2.project_config.get_folders_for_behavior_pipeline()


from imutils.src.imfunctions import stack_z_projection
background_video = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/raw/18072024/EvasRecordings/background/2024-07-18_19-52_background_food_BH/2024-07-18_19-52_background_food_BH_NDTiffStack.tif'
background_img = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/behavior/AVG2024-07-18_19-52_background_food_BH_NDTiffStack.tif'

stack_z_projection(
    str(background_video),
    str(background_img),
    'mean',
    'uint8',
    0,
)


from imutils.src.imfunctions import stack_subtract_background


input_ndtiff = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/raw/18072024/EvasRecordings/2024-07-18_17-53_2per_L4_200_worm1/2024-07-18_17-53_2per_L4_200_worm1_BH'
output_filepath = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/behavior/test.btf'
background_img = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/behavior/AVG2024-07-18_19-52_background_food_BH_NDTiffStack.tif'

stack_subtract_background(input_ndtiff, output_filepath, background_img)


from imutils import MicroscopeDataReader
background_parent_folder  ='/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/raw/13062024/background/2024-06-13_12-35_background_BH/'
_ = MicroscopeDataReader(background_parent_folder, as_raw_tiff=False)








get_ipython().run_line_magic('debug', '')


p.project_config.get_behavior_raw_parent_folder_from_red_fname()


BehaviorCodes.plot_behaviors(p.worm_posture_class.beh_annotation(fluorescence_fps=True, use_manual_annotation=True))


err


from wbfm.utils.general.postures.centerline_classes import get_manual_behavior_annotation_fname
get_manual_behavior_annotation_fname(p.project_config, only_check_relative_paths=True)


from wbfm.utils.general.postures.centerline_classes import parse_behavior_annotation_file
parse_behavior_annotation_file(behavior_fname=p.worm_posture_class.filename_manual_beh_annotation, 
                               template_vector=p.worm_posture_class.template_vector(fluorescence_fps=True))


import pandas as pd
behavior_fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/behavior/AVAL_manual_annotation.csv"
behavior_annotations = pd.read_csv(behavior_fname)
behavior_annotations


p.worm_posture_class.beh_annotation(use_manual_annotation=False, fluorescence_fps=True)


p.worm_posture_class.manual_beh_annotation_already_converted_to_fluorescence_fps


vec = p.worm_posture_class.beh_annotation(use_manual_annotation=True, fluorescence_fps=True, DEBUG=True)
vec


df_traces = p.calc_paper_traces()
p.use_physical_time = True
fig = px.line(df_traces['AVAL'])
p.shade_axis_using_behavior(plotly_fig=fig)
fig.show()


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
p.use_physical_time = False
trigger_class = FullDatasetTriggeredAverages.load_from_project(p, trace_opt=dict(use_paper_options=True))


trigger_class.plot_single_neuron_triggered_average('AVA')


# import napari
# viewer = napari.Viewer()
# viewer.add_image(p.red_data)
# viewer.add_image(p.green_data)
# viewer.add_labels(p.segmentation)


df = p.calc_default_traces(rename_neurons_using_manual_ids=True, use_physical_time=True)


# fig = px.line(df['AVAL'])
# p.shade_axis_using_behavior(plotly_fig=fig)

# fig.show()


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


triggered_avg_class = FullDatasetTriggeredAverages.load_from_project(p, trigger_opt=dict(state=BehaviorCodes.STIMULUS, trigger_on_downshift=False),
                                                                    trace_opt=dict(rename_neurons_using_manual_ids=True))


# triggered_avg_class.plot_events_over_trace('URXR')


fig = px.line(triggered_avg_class.df_traces['BAGL'])
p.shade_axis_using_behavior(plotly_fig=fig, additional_shaded_states=[BehaviorCodes.STIMULUS])
fig.show()


fig = px.line(triggered_avg_class.df_traces['ANTIcorL'])
p.shade_axis_using_behavior(plotly_fig=fig, additional_shaded_states=[BehaviorCodes.STIMULUS])
fig.show()


triggered_avg_class.plot_single_neuron_triggered_average('URXR', show_individual_lines=True)#, xlim=[-5, 15])


list_of_triggered_ind = triggered_avg_class.ind_class.triggered_average_indices()


mat = triggered_avg_class.triggered_average_matrix_from_name('URXR')


# triggered_avg_class.ind_class.nan_points_of_state_before_point(mat.copy(), list_of_triggered_ind,
#                                                               DEBUG=True)


triggered_avg_class.ind_class._get_invalid_states_for_prior_index_removal()


triggered_avg_class.ind_class.behavioral_state


import matplotlib.pyplot as plt
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
neuron_list = ['BAG', 'ANTIcor', 'PQR']
triggered_avg_class.plot_multi_neuron_triggered_average(neuron_list, show_legend=True,
                                                       xlim=[-5, 30])#, ylim=[-0.1, 0.1])
plt.legend()



fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13"
project_data = ProjectData.load_final_project_data_from_config(fname)


from wbfm.utils.general.postures.centerline_classes import get_manual_behavior_annotation_fname


get_manual_behavior_annotation_fname(project_data.project_config)


from enum import Flag, auto

class MyFlags(Flag):
    FOO = auto()
    BAR = auto()

# Check if an enum member is an instance of MyFlags
print(isinstance(MyFlags.FOO, MyFlags))  # Output: True








from scipy.interpolate import RegularGridInterpolator

import numpy as np

def f(x, y, z):

    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)

y = np.linspace(4, 7, 22)

z = np.linspace(7, 9, 33)

xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij')#, sparse=True)

data = f(xg, yg, zg)


data.shape, xg.shape, yg.shape, zg.shape


# # Header

# ## Subheader
