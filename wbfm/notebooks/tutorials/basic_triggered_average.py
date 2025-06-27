#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from wbfm.utils.projects.finished_project_data import ProjectData


# # Step 1: using my project class

# In[2]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# ## Check that the behavior annotation files were found
# 
# If not, you can run an approximate behavioral annotator based on pc1

# In[3]:


project_data_gcamp.worm_posture_class


# ### OPTIONAL, if you need to annotate the behaviors using an alternate pipeline:

# In[4]:



# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
# approximate_behavioral_annotation_using_pc1(project_data_gcamp)


# # Step 2: Load the triggered average object from a project class

# In[22]:


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
# help(FullDatasetTriggeredAverages)


# In[23]:


triggered_class = FullDatasetTriggeredAverages.load_from_project(project_data_gcamp)
triggered_class


# # Step 3: Plotting

# In[24]:


triggered_class.plot_single_neuron_triggered_average('neuron_060')


# ## Triggering to a different behavioral state

# In[12]:


# All states are described in one class
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
# help(BehaviorCodes)


# In[13]:


# Pass in a dictionary to modify the triggered averages

trigger_opt = dict(state=BehaviorCodes.REV)
triggered_class = FullDatasetTriggeredAverages.load_from_project(project_data_gcamp, trigger_opt=trigger_opt)


# In[14]:


# To see more options
help(FullDatasetTriggeredAverages.load_from_project)


# In[15]:


# And even more detailed options
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
help(WormFullVideoPosture.calc_triggered_average_indices)


# In[19]:


triggered_class.plot_single_neuron_triggered_average('neuron_060')


# ## Advanced: plot more than one on the same graph

# In[20]:


ax = triggered_class.plot_single_neuron_triggered_average('neuron_060')
triggered_class.plot_single_neuron_triggered_average('neuron_033', ax=ax, is_second_plot=True,
                                                    color='green')


# In[ ]:




