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


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
import plotly.express as px


# In[3]:


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[5]:


all_projects_immob = load_paper_datasets('immob')
all_projects_gfp = load_paper_datasets('gfp')


# # Same but include gfp and immob: Look at the intrinsic dimensionality using a bunch of methods

# In[6]:


import skdim
from collections import defaultdict
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


# In[7]:



methods = [skdim.id.CorrInt, #skdim.id.DANCo, #skdim.id.ESS, 
           skdim.id.FisherS, #skdim.id.KNN, skdim.id.lPCA, 
           skdim.id.MADA, 
           #skdim.id.MiND_ML, skdim.id.MLE, 
           skdim.id.MOM, skdim.id.TLE, #skdim.id.TwoNN
          ]
method_names = [str(method).split('.')[-1] for method in methods]

all_all_projects = dict(gfp=all_projects_gfp, gcamp=all_projects_gcamp, immob=all_projects_immob)

all_all_dim = []
for proj_type, proj_dict in all_all_projects.items():
    all_dim = {}
    for name, proj in tqdm(proj_dict.items()):
        all_dim[name] = defaultdict()
        for i, m in enumerate(tqdm(methods, leave=False)):
            try:
                model = m()
                data = proj.calc_default_traces(use_paper_options=True)
                gid1 = model.fit(data).dimension_
                all_dim[name][i] = gid1
            except ValueError:
                all_dim[name][i] = np.nan
    df_all_dim = pd.DataFrame(all_dim)
    df_all_dim.index = method_names
    # Dimensions: method = columns, dataset=rows
    df_all_dim = df_all_dim.T
    
    df_all_dim['datatype'] = proj_type
    
    all_all_dim.append(df_all_dim)


# In[8]:


df_all_all_dim = pd.concat(all_all_dim)
df_all_all_dim.columns = [i[:-2] if i != 'datatype' else i for i in df_all_all_dim.columns]
df_all_all_dim.head()


# In[9]:


fig = px.box(df_all_all_dim, points='all', color='datatype', color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig=fig, width_factor=0.5, height_factor=0.3)
fig.update_yaxes(title='Dimensionality', showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)
fig.update_xaxes(title='Estimation method')
fig.update_layout(showlegend=False)

fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'raw_data.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)

fig.show()


# ## For main: pool across methods

# In[11]:


# df_all_all_dim.drop(columns='TwoNN').melt(id_vars='datatype')['datatype'].unique()


# In[12]:


from wbfm.utils.general.utils_paper import data_type_name_mapping

df_dim_combined = df_all_all_dim.melt(id_vars='datatype')
df_dim_combined['datatype'] = df_dim_combined['datatype'].map(data_type_name_mapping())

fig = px.box(df_dim_combined, y='value', #x='datatype',
             color='datatype', color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig=fig, width_factor=0.2, height_factor=0.25)
fig.update_yaxes(title='Estimated<br>dimensionality'), #showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)
fig.update_xaxes(range=[-0.4,0.4])

fig.update_xaxes(title='Dataset')
fig.update_layout(showlegend=False)
# fig.update_traces(width = 0.1)

fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'raw_data_pooled.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)

fig.show()


# ### Instead: each method is a single data point (median across datasets)

# In[25]:


df = df_all_all_dim.reset_index().melt(id_vars=['datatype', 'index'])
# Use custom aggregation to keep dataset name
dct = {
    'number': 'mean',
    'object': lambda col: col.mode() if col.nunique() == 1 else np.nan,
}
groupby_cols = ['index']
dct = {k: v for i in [{col: agg for col in df.select_dtypes(tp).columns.difference(groupby_cols)} for tp, agg in dct.items()] for k, v in i.items()}
df_dim_combined = df.groupby(groupby_cols).agg(**{k: (k, v) for k, v in dct.items()})
df_dim_combined['datatype'] = df_dim_combined['datatype'].map(data_type_name_mapping())

df_dim_combined.head()


# In[27]:


from wbfm.utils.general.utils_paper import data_type_name_mapping

fig = px.box(df_dim_combined, y='value', #x='datatype',
             color='datatype', color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'datatype': ['Freely Moving (GFP)', 'Freely Moving (GCaMP)', 'Immobilized (GCaMP)']})
apply_figure_settings(fig=fig, width_factor=0.2, height_factor=0.25)
fig.update_yaxes(title='Estimated<br>dimensionality'), #showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)
fig.update_xaxes(range=[-0.4,0.4])

fig.update_xaxes(title='Dataset')
fig.update_layout(showlegend=False)
# fig.update_traces(width = 0.1)

fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'raw_data_combined.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)

fig.show()


# In[21]:


df_dim_combined['datatype'].unique()


# # Also calculate the dimensionality of the CCA projection space

# In[12]:


from wbfm.utils.visualization.utils_cca import CCAPlotter


# In[13]:



methods = [skdim.id.CorrInt, #skdim.id.DANCo, #skdim.id.ESS, 
           skdim.id.FisherS, #skdim.id.KNN, skdim.id.lPCA, 
           skdim.id.MADA, 
           #skdim.id.MiND_ML, skdim.id.MLE, 
           skdim.id.MOM, skdim.id.TLE, skdim.id.TwoNN
          ]


# In[14]:


# project_data_gcamp.use_physical_x_axis = True

# cca_plotter = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True))


# In[15]:


# X_r, Y_r, cca = self.calc_cca(n_components=5, binary_behaviors=False)


# In[16]:



cca_dim = {}
for name, proj in tqdm(all_projects_gcamp.items()):
    cca_plotter = CCAPlotter(proj, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True))
    data, _, _ = cca_plotter.calc_cca(n_components=5, binary_behaviors=False)
    cca_dim[name] = defaultdict()
    for i, m in enumerate(tqdm(methods, leave=False)):
        try:
            model = m()
            gid1 = model.fit(data).dimension_
            cca_dim[name][i] = gid1
        except ValueError:
            cca_dim[name][i] = np.nan

df_cca_dim = pd.DataFrame(cca_dim)


# In[8]:


method_names = [str(method).split('.')[-1][:-2] for method in methods]
df_cca_dim.index=method_names
# df_cca_dim.head()


# In[7]:


# fig = px.box(df_cca_dim.T, points='all', color_discrete_map=plotly_paper_color_discrete_map(),
#             title="CCA dimensionality")
# fig.update_yaxes(title='Dimensionality', showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)
# fig.update_xaxes(title='Estimation method')

# apply_figure_settings(fig=fig, width_factor=0.5, height_factor=0.25)

# fig.show()

# fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'cca_space.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# ### Binary dimensionality

# In[19]:



cca_dim_binary = {}
for name, proj in tqdm(all_projects_gcamp.items()):
    cca_plotter = CCAPlotter(proj, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True))
    data, _, _ = cca_plotter.calc_cca(n_components=5, binary_behaviors=True)
    cca_dim_binary[name] = defaultdict()
    for i, m in enumerate(tqdm(methods, leave=False)):
        try:
            model = m()
            gid1 = model.fit(data).dimension_
            cca_dim_binary[name][i] = gid1
        except ValueError:
            cca_dim_binary[name][i] = np.nan
df_cca_dim_binary = pd.DataFrame(cca_dim_binary)


# In[20]:


method_names = [str(method).split('.')[-1][:-2] for method in methods]
df_cca_dim_binary.index=method_names


# In[6]:


# fig = px.box(df_cca_dim_binary.T, points='all', color_discrete_map=plotly_paper_color_discrete_map(),
#             title="CCA dimensionality")
# fig.update_yaxes(title='Dimensionality', showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True, range=[1, 5])
# fig.update_xaxes(title='Estimation method')

# apply_figure_settings(fig=fig, width_factor=1, height_factor=0.25)

# fig.show()

# fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'cca_space_binary.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# ### Both on one graph

# In[23]:


df0 = df_cca_dim.copy().T
df0['behavior type'] = 'Continuous'
df1 = df_cca_dim_binary.copy().T
df1['behavior type'] = 'Discrete'

df_cca_both = pd.concat([df0, df1])


# In[24]:


df_cca_both.head()


# In[26]:


fig = px.box(df_cca_both, points='all', color='behavior type')

apply_figure_settings(fig=fig, width_factor=0.5, height_factor=0.3)
fig.update_yaxes(title='Dimensionality', showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True, range=[1, 5])
fig.update_xaxes(title='Estimation method')

fig.show()

fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'cca_space_both.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[32]:


# fig = px.scatter(df_cca_both, color='behavior type', marginal_y='box')

# apply_figure_settings(fig=fig, width_factor=1, height_factor=0.5)
# fig.update_yaxes(title='Dimensionality', showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True, range=[1, 5])
# fig.update_xaxes(title='Estimation method')

# fig.show()

# fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/intrinsic_dimension", 'cca_space_both2.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# # Scratch

# In[ ]:





# # Look at the intrinsic dimensionality using a bunch of methods

# In[6]:


import skdim
from collections import defaultdict


# In[ ]:



# methods = [skdim.id.CorrInt, #skdim.id.DANCo, #skdim.id.ESS, 
#            skdim.id.FisherS, skdim.id.KNN, skdim.id.lPCA, skdim.id.MADA, skdim.id.MiND_ML, skdim.id.MLE, skdim.id.MOM, skdim.id.TLE, skdim.id.TwoNN
#           ]

# all_dim = {}
# for name, proj in tqdm(all_projects_gcamp.items()):
#     all_dim[name] = defaultdict()
#     for i, m in enumerate(tqdm(methods, leave=False)):
#         try:
#             model = m()
#             data = proj.calc_default_traces(use_paper_options=True)
#             gid1 = model.fit(data).dimension_
#             all_dim[name][i] = gid1
#         except ValueError:
#             all_dim[name][i] = np.nan


# In[ ]:


# method_names = [str(method).split('.')[-1] for method in methods]


# In[ ]:


# df_all_dim = pd.DataFrame(all_dim)
# df_all_dim.index=method_names


# In[ ]:


# px.box(df_all_dim.T, points='all')

