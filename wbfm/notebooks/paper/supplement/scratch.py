#!/usr/bin/env python
# coding: utf-8

# In[4]:


from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
import plotly.express as px
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
import napari
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


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


# In[6]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


# In[7]:


# Same individual: fm and immob
fname = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28'
project_data_fm2immob_fm = ProjectData.load_final_project_data_from_config(fname, verbose=0)


# In[8]:


from wbfm.utils.projects.finished_project_data import plot_pca_projection_3d_from_project


# In[ ]:


plot_pca_projection_3d_from_project(project_data_fm2immob_fm, include_time_series_subplot=False)


# In[78]:


from wbfm.utils.visualization.utils_export_videos import save_video_of_pca_plot_with_behavior
save_video_of_pca_plot_with_behavior(project_data_fm2immob_fm, t_max=100)


# In[75]:





# In[65]:


import matplotlib.pyplot as plt
import numpy as np
import time

# Turn on interactive mode
plt.ion()

# Initialize the figure and axis
fig, ax = plt.subplots()

# Set up the initial data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
(line,) = ax.plot(x, y, label="Sin Wave")

# Set limits and labels
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# Update loop
for i in range(100):
    # Shift the sine wave
    y = np.sin(x + i * 0.1)
    
    # Update the line data
    line.set_data(x, y)
    
    # Redraw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Pause for a short duration to see the animation
    time.sleep(0.001)
    plt.pause(0.001)

plt.ioff()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


df_traces = project_data_fm2immob_fm.calc_paper_traces(interpolate_nan=True)
df_traces_nan = project_data_fm2immob_fm.calc_paper_traces(interpolate_nan=False)


# In[ ]:





# In[3]:


import numpy as np
x = [0.5, 0.6, 0.8, 0.95, 1.0]
xd = np.diff(x)
np.insert(xd, 0, x[0])


# In[ ]:





# In[ ]:





# In[16]:


px.imshow(df_traces.T)


# In[17]:


px.imshow(df_traces_nan.T)


# In[ ]:





# In[ ]:





# In[18]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_O2_immob = load_paper_datasets('immob_o2')


# In[24]:


all_projects_O2_immob_mutant = load_paper_datasets('hannah_O2_immob_mutant')


# In[19]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, plot_box_multi_axis
from wbfm.utils.visualization.paper_multidataset_triggered_average import plot_ttests_from_triggered_average_classes, plot_triggered_averages_from_triggered_average_classes
from wbfm.utils.external.utils_plotly import combine_plotly_figures
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[20]:


opt = dict(calculate_residual=False, calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# Immob
triggered_average_gcamp_plotter_immob = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                         trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6),
                                                                         calculate_stimulus=True)

triggered_average_gcamp_plotter_immob_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                                    calculate_stimulus=True,
                                                                                    trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6))


# In[25]:


# Immob mutant
triggered_average_gcamp_plotter_immob_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, **opt, 
                                                                                 calculate_stimulus=True, 
                                                                                 trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6))


# In[26]:


# summary_function = lambda x, **kwargs: np.nanquantile(x, 0.99, **kwargs)
# summary_function = np.nanmax
summary_function = None
neuron_list = ['AQR', 'URX', 'AUA', 'BAG', 'ANTIcor', 'RMDV', 'IL1L', 'IL2L']

# output_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant_with_ttests'
opt = dict(show_title=True, output_dir=None, min_lines=1, xlim=[-5, 12], show_individual_lines=False,
           round_y_ticks=False, show_x_label=True, show_y_ticks=True, show_y_label=False, i_figure=4,  
           use_plotly=True, to_show=False)
##
## IMMOB
##
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]
is_mutant_vec = [False, True]
output_dir = None
all_figs_trig, df_boxplot, df_p_values = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=output_dir, to_show=False,
                                                                                   ttest_kwargs=dict(dynamic_window_center=True, DEBUG=True))
title = 'Stimulus'
all_figs_box = plot_triggered_averages_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, trigger_type, **opt)

# Combine and actually plot
_combine_and_save(all_figs_box, all_figs_trig, all_figs_examples=None, suffix='-upshift')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd


# In[ ]:


fname = '/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/projects/19062024/2024-06-19_16-08_mec4_7b_higher_red_worm4_tap-2024-06-19/behavior/raw_stack_AVG_background_subtracted_normalisedDLC_resnet50_wbfm_nose_tailJan4shuffle1_1030000.h5'
df = pd.read_hdf(fname)


# In[4]:


# fname = "/home/charles/Current_work/collaboration/konstantinos/adult_1030.nd2"
# import nd2
# img = nd2.imread(fname)
# v = napari.view_image(img)


# In[4]:


img.shape


# In[ ]:





# In[2]:


# all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')
# 


# In[ ]:





# In[9]:


# fname = "/lisc/scratch/neurobiology/zimmer/wbfm/test_projects/freely_moving/pytest-raw/project_config.yaml"
fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-41_ZIM2165_immob_worm5-2022-12-06'
p = ProjectData.load_final_project_data_from_config(fname)

# p.project_config.get_folders_for_behavior_pipeline()


# In[11]:


p.neuron_name_to_manual_id_mapping(confidence_threshold=0, remove_unnamed_neurons=True, remove_duplicates=False)


# In[5]:


# fname = "/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/projects/13062024/2024-06-13_17-42_control_worm10_wflyback-2024-06-13/project_config.yaml"
fname = "/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/project_config.yaml"
p2 = ProjectData.load_final_project_data_from_config(fname)

# p = all_projects_O2_immob['2023-09-07_16-11_CaMP7b_O2_worm1-2023-09-07']
# p.use_physical_time = True


# In[9]:


p2.project_config.get_folders_for_behavior_pipeline()


# In[15]:


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


# In[10]:


from imutils.src.imfunctions import stack_subtract_background


# In[17]:


input_ndtiff = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/raw/18072024/EvasRecordings/2024-07-18_17-53_2per_L4_200_worm1/2024-07-18_17-53_2per_L4_200_worm1_BH'
output_filepath = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/behavior/test.btf'
background_img = '/lisc/scratch/neurobiology/zimmer/EvaGratzl/WBFM/projects/18072024_Eva_only/2024-07-18_17-53_2per_L4_200_worm1-2024-07-18/behavior/AVG2024-07-18_19-52_background_food_BH_NDTiffStack.tif'

stack_subtract_background(input_ndtiff, output_filepath, background_img)


# In[19]:


from imutils import MicroscopeDataReader
background_parent_folder  ='/lisc/scratch/neurobiology/zimmer/Pidde/WBFM/raw/13062024/background/2024-06-13_12-35_background_BH/'
_ = MicroscopeDataReader(background_parent_folder, as_raw_tiff=False)


# In[ ]:





# In[ ]:





# In[8]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


p.project_config.get_behavior_raw_parent_folder_from_red_fname()


# In[5]:


BehaviorCodes.plot_behaviors(p.worm_posture_class.beh_annotation(fluorescence_fps=True, use_manual_annotation=True))


# In[ ]:


err


# In[ ]:


from wbfm.utils.general.postures.centerline_classes import get_manual_behavior_annotation_fname
get_manual_behavior_annotation_fname(p.project_config, only_check_relative_paths=True)


# In[ ]:


from wbfm.utils.general.postures.centerline_classes import parse_behavior_annotation_file
parse_behavior_annotation_file(behavior_fname=p.worm_posture_class.filename_manual_beh_annotation, 
                               template_vector=p.worm_posture_class.template_vector(fluorescence_fps=True))


# In[ ]:


import pandas as pd
behavior_fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/behavior/AVAL_manual_annotation.csv"
behavior_annotations = pd.read_csv(behavior_fname)
behavior_annotations


# In[ ]:


p.worm_posture_class.beh_annotation(use_manual_annotation=False, fluorescence_fps=True)


# In[ ]:


p.worm_posture_class.manual_beh_annotation_already_converted_to_fluorescence_fps


# In[ ]:


vec = p.worm_posture_class.beh_annotation(use_manual_annotation=True, fluorescence_fps=True, DEBUG=True)
vec


# In[ ]:


df_traces = p.calc_paper_traces()
p.use_physical_time = True
fig = px.line(df_traces['AVAL'])
p.shade_axis_using_behavior(plotly_fig=fig)
fig.show()


# In[ ]:


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
p.use_physical_time = False
trigger_class = FullDatasetTriggeredAverages.load_from_project(p, trace_opt=dict(use_paper_options=True))


# In[ ]:


trigger_class.plot_single_neuron_triggered_average('AVA')


# In[ ]:


# import napari
# viewer = napari.Viewer()
# viewer.add_image(p.red_data)
# viewer.add_image(p.green_data)
# viewer.add_labels(p.segmentation)


# In[ ]:


df = p.calc_default_traces(rename_neurons_using_manual_ids=True, use_physical_time=True)


# In[ ]:


# fig = px.line(df['AVAL'])
# p.shade_axis_using_behavior(plotly_fig=fig)

# fig.show()


# In[ ]:


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[ ]:


triggered_avg_class = FullDatasetTriggeredAverages.load_from_project(p, trigger_opt=dict(state=BehaviorCodes.STIMULUS, trigger_on_downshift=False),
                                                                    trace_opt=dict(rename_neurons_using_manual_ids=True))


# In[ ]:


# triggered_avg_class.plot_events_over_trace('URXR')


# In[ ]:


fig = px.line(triggered_avg_class.df_traces['BAGL'])
p.shade_axis_using_behavior(plotly_fig=fig, additional_shaded_states=[BehaviorCodes.STIMULUS])
fig.show()


# In[ ]:


fig = px.line(triggered_avg_class.df_traces['ANTIcorL'])
p.shade_axis_using_behavior(plotly_fig=fig, additional_shaded_states=[BehaviorCodes.STIMULUS])
fig.show()


# In[ ]:


triggered_avg_class.plot_single_neuron_triggered_average('URXR', show_individual_lines=True)#, xlim=[-5, 15])


# In[ ]:


list_of_triggered_ind = triggered_avg_class.ind_class.triggered_average_indices()


# In[ ]:


mat = triggered_avg_class.triggered_average_matrix_from_name('URXR')


# In[ ]:


# triggered_avg_class.ind_class.nan_points_of_state_before_point(mat.copy(), list_of_triggered_ind,
#                                                               DEBUG=True)


# In[ ]:


triggered_avg_class.ind_class._get_invalid_states_for_prior_index_removal()


# In[ ]:


triggered_avg_class.ind_class.behavioral_state


# In[ ]:


import matplotlib.pyplot as plt
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
neuron_list = ['BAG', 'ANTIcor', 'PQR']
triggered_avg_class.plot_multi_neuron_triggered_average(neuron_list, show_legend=True,
                                                       xlim=[-5, 30])#, ylim=[-0.1, 0.1])
plt.legend()


# In[ ]:



fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13"
project_data = ProjectData.load_final_project_data_from_config(fname)


# In[ ]:


from wbfm.utils.general.postures.centerline_classes import get_manual_behavior_annotation_fname


# In[ ]:


get_manual_behavior_annotation_fname(project_data.project_config)


# In[ ]:


from enum import Flag, auto

class MyFlags(Flag):
    FOO = auto()
    BAR = auto()

# Check if an enum member is an instance of MyFlags
print(isinstance(MyFlags.FOO, MyFlags))  # Output: True


# In[ ]:





# In[ ]:





# In[20]:


from scipy.interpolate import RegularGridInterpolator

import numpy as np

def f(x, y, z):

    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)

y = np.linspace(4, 7, 22)

z = np.linspace(7, 9, 33)

xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij')#, sparse=True)

data = f(xg, yg, zg)


# In[21]:


data.shape, xg.shape, yg.shape, zg.shape


# # Header

# ## Subheader
