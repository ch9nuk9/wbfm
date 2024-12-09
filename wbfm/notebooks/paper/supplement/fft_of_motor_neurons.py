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
import plotly.express as px


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


# In[3]:


from wbfm.utils.general.hardcoded_paths import load_all_data_as_dataframe


# In[4]:


Xy = load_all_data_as_dataframe()


# In[5]:


Xy.head()


# # FFT of VB02 in immob and freely moving

# In[6]:


from wbfm.utils.projects.finished_project_data import plot_frequencies_for_fm_and_immob_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, apply_figure_settings, data_type_name_mapping
from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading


# In[30]:


output_folder = 'multiplexing'


# In[40]:


from scipy import signal
fs = 3.5


# In[66]:


def package_df_for_fft(neuron_name):

    def func(y): 
        output = signal.welch(y, fs)
        pxx = np.array(output[1])
        # pxx = (pxx - np.mean(pxx)) / np.std(pxx)
        pxx = (pxx) / np.std(pxx)
        return pd.Series({'freq': output[0], 'pxx': pxx})

    pxx_vb02 = Xy.groupby(['dataset_name', 'dataset_type'])[neuron_name].apply(func).to_frame()
    
    pxx_vb02_explode = pxx_vb02.unstack(level=2).apply(pd.Series.explode).droplevel(level=0, axis=1).reset_index(level=1)

    _df = pxx_vb02_explode.copy()
    _df = _df[_df['dataset_type'] != 'gfp']
    _df['dataset_type'] = _df['dataset_type'].map(data_type_name_mapping())
    
    return _df.dropna()


# In[67]:


df.head()


# In[71]:


px.line(df[df['dataset_type'] == 'Freely Moving (GCaMP)'].reset_index(), x='freq', y='pxx', color='dataset_name',)


# In[70]:


# Do not plot gfp

neurons_to_plot = ['VB02', 'VB03', 'VB01', 'DB01', 'DB02', 'VA01', 'VA02', 'DA01',
                  'SIAVL', 'SIAVR', 'DD01', 'RMDDR', 'RMDDL', 'RMDVL', 'RMDVR', 'SMDDL', 'SMDDR', 
                   'RMEV', 'RMED', 'RMER', 'RMEL', 'AVFL', 'AVFR', 'AVAL', 'AVAR']

for n in neurons_to_plot:

    df = package_df_for_fft(n)
    print(n)

    fig = plotly_plot_mean_and_shading(df, x='freq', y='pxx', color='dataset_type',
                                      cmap = plotly_paper_color_discrete_map(), title=n)
    fig.update_xaxes(range=[0, 0.4], title='Frequency (Hz)')
    fig.update_yaxes(title='Normalized<br> power')
    fig.update_layout(
        showlegend=(n=='VB02'),
        legend=dict(
          yanchor="top",
          y=1.0,
          xanchor="left",
          x=0.25
        ),
    )

    apply_figure_settings(fig, width_factor=0.4, height_factor=0.15)

    fig.show()

    output_foldername = 'fft_examples'
    fname = os.path.join(output_foldername, f'fft_neuron_{n}.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    
    break


# In[24]:


# # Do not plot gfp

# df = package_df_for_fft('DB01')

# fig = plotly_plot_mean_and_shading(df, x='freq', y='pxx', color='dataset_type',
#                                   cmap = plotly_paper_color_discrete_map())
# fig.update_xaxes(range=[0, 0.5], title='Frequency (Hz)')
# fig.update_yaxes(title='Normalized power')
# fig.update_layout(showlegend=False)

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)

# fig.show()

# # output_foldername = 'intro/dimensionality'
# # fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative.png')
# # fig.write_image(fname, scale=3)
# # fname = Path(fname).with_suffix('.svg')
# # fig.write_image(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# # Alternative

# ## Sanity check: fft of small vb02 time series

# In[73]:


Xy['dataset_name'].unique()


# In[76]:


y = Xy[Xy['dataset_name']=='ZIM2165_Gcamp7b_worm1-2022_11_28']['VB02'].reset_index(drop=True)
px.line(y)


# In[79]:


y_subset = y.iloc[900:1100].values
output = signal.welch(y_subset, fs)
pxx = np.array(output[1])
# pxx = (pxx - np.mean(pxx)) / np.std(pxx)
# pxx = (pxx) / np.std(pxx)

px.line(y=pxx, x=output[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Use projects

# In[74]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[75]:


all_projects_gfp = load_paper_datasets('gfp')


# In[76]:


all_projects_immob = load_paper_datasets('immob')


# In[ ]:



opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'VB02', 
                                                                                                    output_folder=output_folder,**opt)


# In[ ]:



opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'AVAL', 
                                                                                                    output_folder=output_folder, **opt)


# In[ ]:





# In[ ]:




