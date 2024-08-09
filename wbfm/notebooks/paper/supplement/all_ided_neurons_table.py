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


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
import plotly.express as px
from wbfm.utils.general.utils_filenames import add_name_suffix


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
# all_projects_gfp = load_paper_datasets('gfp')


all_projects_immob = load_paper_datasets('immob')


all_projects_immob_O2 = load_paper_datasets('hannah_O2_immob')


# # Freely moving: ID'ed neurons

all_ids = defaultdict(int)

for name, proj in tqdm(all_projects_gcamp.items()):
    df_traces = proj.calc_default_traces(use_paper_options=True)
    neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
    for c in neuron_columns:
        all_ids[c] += 1


df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
px.scatter(df_ids)


fname = os.path.join('ids', 'id_counts.csv')
df_ids.to_csv(fname)


# # Immobilized: ID'ed neurons

all_ids = defaultdict(int)

for name, proj in tqdm(all_projects_immob.items()):
    df_traces = proj.calc_default_traces(use_paper_options=True)
    neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
    for c in neuron_columns:
        all_ids[c] += 1


df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
px.scatter(df_ids)


fname = os.path.join('ids', 'id_counts_immob.csv')
df_ids.to_csv(fname)


# # Immobilized with O2 stimulus: ID'ed neurons

all_ids = defaultdict(int)

for name, proj in tqdm(all_projects_immob_O2.items()):
    df_traces = proj.calc_default_traces(use_paper_options=True)
    neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
    for c in neuron_columns:
        all_ids[c] += 1


df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
px.scatter(df_ids)


fname = os.path.join('ids', 'id_counts_immob_O2.csv')
df_ids.to_csv(fname)


# # Same but with simple dataframes, not full projects

from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir
# fname = 'data.h5'

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
Xy_gfp = pd.read_hdf(fname)


def get_counts(Xy):
    all_ids = np.ceil(Xy.count() / Xy.shape[0] * len(Xy['dataset_name'].unique()))
    strs_to_drop = ['neuron', 'eigenworm', 'pca', 'name', 'time', 'curvature', 'manifold',
                   'fwd', 'speed', 'collision']
    for s in strs_to_drop:
        all_ids = all_ids[~all_ids.index.str.contains(s)]
    all_ids = all_ids.sort_values()
    return all_ids

all_ids = get_counts(Xy)
all_ids_gfp = get_counts(Xy_gfp)


px.scatter(all_ids_gfp)








# # Scratch: Do any projects have no IDs?

all_num_ids = defaultdict(int)

for name, proj in tqdm(all_projects_gcamp.items()):
    df_traces = proj.calc_default_traces(use_paper_options=True)
    neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
    if 'RMED' not in neuron_columns:
        print(name)
    if 'AQ' in neuron_columns:
        print(name)
    all_num_ids[name] = len(neuron_columns)


p = all_projects_gcamp['ZIM2165_Gcamp7b_worm2-2022-12-10']
df = p.calc_default_traces(use_paper_options=True)


all_num_ids = defaultdict(int)

for name, proj in tqdm(all_projects_immob.items()):
    df_traces = proj.calc_default_traces(use_paper_options=True)
    neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
    if 'RMED' not in neuron_columns:
        print(name)
    if 'AQ' in neuron_columns:
        print(name)
    all_num_ids[name] = len(neuron_columns)
min(list(all_num_ids.values()))


all_num_ids




