#!/usr/bin/env python
# coding: utf-8

# In[ ]:





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


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
import plotly.express as px
from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets


# In[3]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


# Load multiple datasets
from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets('gcamp')


# In[5]:


all_projects_gfp = load_paper_datasets('gfp')


# In[6]:


all_projects_immob = load_paper_datasets('immob')


# In[7]:


all_projects_good = load_paper_datasets('gcamp_good')


# # Load traces from all datasets, with manual names

# In[7]:


from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects, build_behavior_time_series_from_multiple_projects
from wbfm.utils.general.utils_behavior_annotation import shade_using_behavior_plotly, shade_stacked_figure_using_behavior_plotly
from wbfm.utils.general.utils_behavior_annotation import plot_stacked_figure_with_behavior_shading_using_plotly
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['AVAR', 'AVAL', 'AVEL', 'AVER', 'URYVL', 'URYVR', 'signed_stage_speed'])
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['AVAR', 'AVAL', 'signed_stage_speed'])
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['hesitation', 'head_cast', 'RIS', 'signed_stage_speed'])
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RIVL', 'RIVR', 'RID', 'BAGL', 'BAGR'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RIVL', 'RIVR', 'SMDVL', 'SMDVR'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[34]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['AVBL', 'AVBR', 'RID', 'RMEL', 'RMER', 'RMED', 'RMEV', 'RIS'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['AVBL', 'AVBR', 'RID', 'BAGR', 'BAGL', 'AVAL', 'AVAR'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['VG_middle_ramping_L', 'VG_middle_ramping_R', 'RIVL', 'RIVR', 'VB02', 'AVAL'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['VG_post_turning_R', 'VG_post_turning_L', 'RIVL', 'RIVR', 'VB02'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['VG_post_turning_R', 'VG_post_turning_L', 'VG_middle_ramping_L', 'VG_middle_ramping_R', 
                                                                                              'VG_anter_FWD_no_curve_R', 'VG_anter_FWD_no_curve_L', 'VG_post_FWD_L', 'VG_post_FWD_R'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RIVR', 'RIVL', 'RID'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['VB01', 'VB02', 'VB03', 'VA01', 'VA02', 'DB01', 'DB02', 'DA01'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['BAGL', 'BAGR'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['VB02', 'DB01', 'DB02'], DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'RIS', 'AVBL', 'AVBR'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'VB02', 'AVAL'],
                                                            trace_kwargs=dict(residual_mode='pca', interpolate_nan=True))
fig.show()


# ### Look at the forward neurons, with and without O2 sensory

# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'AVBL', 'AVBR', 'RIBL', 'RIBR'], DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'BAGL', 'BAGR', 'IL2LL', 'IL1LL', 'IL2LR', 'IL1LR'], DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'RIVL', 'RIVR', 'SMDDL', 'SMDDR', 'VG_post_turning_L', 'VG_post_turning_R'],
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['RID', 'RIVL', 'RIVR', 'BAGL', 'BAGR'], DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, column_names=['AVAL', 'RIVL', 'VB01', 'SMDVR'], DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, column_names=['VB02', 'RMEV', 'RMED'],
                                                            trace_kwargs=dict(channel_mode='red'))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, column_names=['AVAL', 'AVAR'],
                                                            trace_kwargs=dict(channel_mode='red'))
fig.show()


# # Same but for ALL datasets (note: few IDs in these)

# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, column_names=['VB02', 'RMEV', 'RMED'])
fig.show()


# # Same for gfp

# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gfp, column_names=['VB02', 'RMEV', 'RMED'])
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gfp, column_names=['VB02', 'RMEV', 'RMED'],
                                                            trace_kwargs=dict(channel_mode='red'))
fig.show()


# # Same but for immob

# In[15]:


from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_ava


# In[16]:


for name, p in all_projects_immob.items():
    approximate_behavioral_annotation_using_ava(p)


# In[21]:


# # Force reload of behavior
# for name, p in all_projects_immob.items():
#     # del all_projects_immob.__dict__['worm_posture_class']
#     all_projects_immob.worm_posture_class.cache_clear()


# In[ ]:


# fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RMEV', 'RMED', 'SMDVL'], fname_suffix='immob',
#                                                             trace_kwargs=dict(channel_mode='red'))
# fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['AVAL', 'VB01'], fname_suffix='immob', DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['AVAL', 'VB01', 'VB02', 'VB03'], fname_suffix='immob', DEBUG=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['BAGR', 'BAGL', 'AVAL'], fname_suffix='immob')
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['DB01', 'VB01', 'VB02', ], fname_suffix='immob')
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['AVAL', 'RID', 'RIVL', 'RIVR'], fname_suffix='immob')
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['AVAL', 'RIVL', 'AVBL', 'SMDVL', 'SMDDL'], fname_suffix='immob')
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RID', 'RIS', 'AVBL', 'AVBR', 'AVAL', 'AVAR'], fname_suffix='immob',
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[8]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RIS', 'AVBL', 'AVBR', 'VB02'], fname_suffix='immob',
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RID', 'RIS', 'RIVL', 'RIVR', 'SMDDL', 'SMDDR', 'AVAL', 'AVAR'], fname_suffix='immob',
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RID', 'RIVR', 'SMDDL', 'SMDVL', 'SMDVR', 'AVAL', 'RIS'], fname_suffix='immob',
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['RID', 'RIS', 'RMEL', 'RMEV', 'AVBL', 'AVBR', 'RIBL', 'RIBR', 'AVAL', 'AVAR'], fname_suffix='immob',
                                                            trace_kwargs=dict(manual_id_confidence_threshold=0))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, column_names=['AVAL', 'VB02'], fname_suffix='immob', DEBUG=False,
                                                            trace_kwargs=dict(residual_mode='pca', interpolate_nan=True))
fig.show()


# # Scratch

# In[30]:


p = all_projects_good['ZIM2165_Gcamp7b_worm1-2022_11_28']
# df_traces = p.calc_default_traces(rename_neurons_using_manual_ids=True)
kymo = p.worm_posture_class.hilbert_frequency(fluorescence_fps=True, reset_index=True)


# In[31]:


# fig = px.line(df_traces['AVAR'])
# shade_using_behavior_plotly(beh_vec, fig, DEBUG=True)
# fig.show()


# In[ ]:


px.imshow(kymo.T, aspect=5, zmin=-0.1, zmax=0.5)


# In[ ]:


px.imshow(kymo.T.abs(), aspect=5, zmin=-0.1, zmax=0.5)


# In[ ]:




