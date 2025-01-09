#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


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
from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
import plotly.express as px


# # First load multiple projects as a dictionary
# 
# I have convenience functions for the paper datasets

# In[16]:


# Load multiple datasets
all_projects_good = load_paper_datasets('gcamp_good')


# In[17]:


all_projects_immob = load_paper_datasets('immob')


# # Use a function to plot traces across multiple projects (with proper behavior shading)

# In[12]:


from wbfm.utils.general.utils_behavior_annotation import plot_stacked_figure_with_behavior_shading_using_plotly


# ## Plot

# In[35]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, ['AVBL', 'AVBR', 'RIS'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# ## Plot: immobilized

# In[37]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'AVBR', 'RIS'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# In[ ]:




