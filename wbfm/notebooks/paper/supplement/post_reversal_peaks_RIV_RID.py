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
import plotly.express as px


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# # Look at RID in an example dataset, coloring by post-reversal self-collision

opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean')

df_traces = project_data_gcamp.calc_default_traces(**opt)

idx_rid = project_data_gcamp.neuron_name_to_manual_id_mapping(flip_names_and_ids=True)['RID']
y_rid = df_traces[idx_rid]


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
worm = project_data_gcamp.worm_posture_class
beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)

y_rev = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.REV).astype(int)
y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)


plt.figure(dpi=200, figsize=(10, 3))
plt.plot(y_rid)
plt.plot(df_traces['neuron_045'])
project_data_gcamp.shade_axis_using_behavior(additional_shaded_states=[BehaviorCodes.DORSAL_TURN])


df = pd.DataFrame({'y_rid': y_rid, 'y_rev': y_rev, 'y_collision': y_collision})

px.line(df)


peaks_rid = worm.get_peaks_post_reversal(y_rid, allow_reversal_before_peak=True)
peaks_collision = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)


# df = pd.DataFrame({'peak_rid': peaks_rid, 'collision': peaks_collision})
# px.box(df, x='collision', y='peak_rid')


# # Loop over all datasets (RID should be ID'ed)

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
# opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean')

all_peaks_rid = {}
all_collision_flags = {}
for name, p in tqdm(all_projects_gcamp.items()):
    try:
        df_traces = p.calc_paper_traces()
        idx_rid = p.neuron_name_to_manual_id_mapping(flip_names_and_ids=True, confidence_threshold=0)['RID']
        y_rid = df_traces[idx_rid]

        worm = p.worm_posture_class

        beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)

        all_peaks_rid[name] = worm.get_peaks_post_reversal(y_rid, allow_reversal_before_peak=True)
        all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)
    except (ValueError, KeyError):
        continue


# Melt these dataframes into one with a single column
df_rid = pd.DataFrame.from_dict(all_peaks_rid, orient='index').T.stack().dropna().reset_index(level=1)
df_rid.columns = ['dataset_name', 'rid_peaks']

df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().dropna().reset_index(level=1, drop=True)


df = pd.concat([df_rid, df_collision.astype(int)], axis=1)
df.columns = ['dataset_name', 'rid_peaks', 'collision_flag']


# px.box(df, x='collision_flag', y='rid_peaks', color='collision_flag')


# px.box(df, x='collision_flag', y='rid_peaks', color='dataset_name')


# # Correlate RID and curvature post reversals

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.utils_paper import apply_figure_settings


# opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean')

all_peaks_rid = {}
all_collision_flags = {}
all_curvature_peaks = {}
all_curvature_dorsal_peaks = {}
all_ventral_states = {}
all_dorsal_states = {}
for name, p in tqdm(all_projects_gcamp.items()):
    try:
        df_traces = p.calc_paper_traces()
        # idx_rid = p.neuron_name_to_manual_id_mapping(flip_names_and_ids=True, confidence_threshold=0)['RID']
        if 'RID' not in df_traces:
            continue
        y_rid = df_traces['RID']

        worm = p.worm_posture_class

        beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)
        y_ventral_turn = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.VENTRAL_TURN).astype(int)
        y_dorsal_turn = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.DORSAL_TURN).astype(int)
        
        # y_curvature = worm.calc_behavior_from_alias('summed_signed_curvature', reset_index=True)
        y_curvature = worm.calc_behavior_from_alias('ventral_only_head_curvature')
        y_curvature_dorsal = worm.calc_behavior_from_alias('dorsal_only_head_curvature')
        # y_curvature = worm.calc_behavior_from_alias('interpolated_ventral_minus_dorsal_midbody_curvature', reset_index=True)

        all_peaks_rid[name] = worm.get_peaks_post_reversal(y_rid, allow_reversal_before_peak=True)[0]
        all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)[0]
        all_curvature_peaks[name] = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)[0]
        all_curvature_dorsal_peaks[name] = worm.get_peaks_post_reversal(y_curvature_dorsal, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)[0]
        all_ventral_states[name] = worm.get_peaks_post_reversal(y_ventral_turn, allow_reversal_before_peak=False)[0]
        all_dorsal_states[name] = worm.get_peaks_post_reversal(y_dorsal_turn, allow_reversal_before_peak=False)[0]
    except (ValueError, KeyError) as e:
        print(f"Error on dataset: {name}")
        # continue
        raise e


# %debug


# Melt these dataframes into one with a single column
df_rid = pd.DataFrame.from_dict(all_peaks_rid, orient='index').T.stack().reset_index()
df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=True)
df_curvature = pd.DataFrame.from_dict(all_curvature_peaks, orient='index').T.stack().reset_index(drop=True)
df_curvature_dorsal = pd.DataFrame.from_dict(all_curvature_dorsal_peaks, orient='index').T.stack().reset_index(drop=True)
df_vt = pd.DataFrame.from_dict(all_ventral_states, orient='index').T.stack().reset_index(drop=True)
df_dt = pd.DataFrame.from_dict(all_dorsal_states, orient='index').T.stack().reset_index(drop=True)

df_list = [df_rid, df_collision.astype(bool), df_curvature, df_curvature_dorsal, df_vt, df_dt]
df = pd.concat(df_list, join='inner', axis=1)
df.columns = ['dataset_idx', 'dataset_name', 'rid_peaks', 'collision_flag', 'curvature_ventral', 'curvature_dorsal', 'vt', 'dt']

df['ventral_turn'] = df['curvature_ventral'] > 0
# Create a mini-enum for the 4 possibilities
df['post_reversal_turn'] = df['vt'] + 2*df['dt']


# df_rid.shape, df.shape, df_vt.shape, df_curvature_dorsal.shape


# sum(map(len, list(all_peaks_rid.values()))), sum(map(len, list(all_curvature_dorsal_peaks.values())))


# fig = px.scatter(df[df['curvature_ventral']>0], x='curvature_ventral', y='rid_peaks', trendline='ols', 
#            title="Correlation between amplitude of post-reversal RID peak and ventral curvature peak")

# # apply_figure_settings(fig, width_factor=1.0, 


# px.scatter(df, x='curvature_ventral', y='rid_peaks', color='collision_flag', trendline='ols')


# px.scatter(df, x='curvature_ventral', y='rid_peaks', color='ventral_turn', trendline='ols')


# px.scatter(df, x='curvature_dorsal', y='rid_peaks', color='ventral_turn', trendline='ols')


# fig = px.scatter(df, x='curvature_ventral', y='rid_peaks', facet_row='post_reversal_turn', trendline='ols', height=1000)
# fig.show()


# # Same but for SMD and RIV (as many as found)

from wbfm.utils.general.utils_paper import  apply_figure_settings
from wbfm.utils.external.utils_plotly import add_trendline_annotation


from collections import defaultdict
# opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean')

names_to_check = [#'SMDDR', 'SMDDL', 
                  'SMDVL', 'SMDVR', 'RIVL', 'RIVR', 'RID']

all_peaks_named_neurons = defaultdict(dict)
all_collision_flags = {}

curvature_aliases = ['ventral_only_head_curvature', 'summed_signed_curvature', 'ventral_quantile_curvature', 'head_signed_curvature', 'ventral_only_body_curvature']
all_curvature_peaks_dict = {a: {} for a in curvature_aliases}
# all_curvature_peaks = {}
for name, p in tqdm(all_projects_gcamp.items()):
    try:
        df_traces = p.calc_paper_traces()
        y_neurons = {n: df_traces[n] for n in names_to_check if n in df_traces}
                
        worm = p.worm_posture_class

        beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)
        
        # y_curvature = worm.calc_behavior_from_alias('summed_signed_curvature', reset_index=True)
        # y_curvature = worm.calc_behavior_from_alias('ventral_only_head_curvature')
        
        all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)[0]
        # curv_peaks = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)[0]
        # all_curvature_peaks[name] = curv_peaks
        
        for alias in all_curvature_peaks_dict.keys():
            y_curvature = worm.calc_behavior_from_alias(alias)
            curv_peaks = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)[0]
            all_curvature_peaks_dict[alias][name] = curv_peaks

        # Even if the neurons aren't found, the lists must be populated to be the same length as all others
        for _name in names_to_check:
            y = y_neurons.get(_name, None)
            if y is None:
                all_peaks_named_neurons[_name][name] = [np.nan]*len(curv_peaks)
                # all_peaks_named_neurons[name][_name] = [np.nan]*len(curv_peaks)
            else:
                all_peaks_named_neurons[_name][name] = worm.get_peaks_post_reversal(y, allow_reversal_before_peak=True)[0]
                # all_peaks_named_neurons[name][_name] = worm.get_peaks_post_reversal(y, allow_reversal_before_peak=True)[0]
                # print(f"Found {_name} in dataset {name}")

    except (ValueError, KeyError) as e:
        print(f"Error on dataset: {name}")
        print(e)
        # raise e


my_reshape = lambda d: pd.DataFrame.from_dict(d, orient='index').T.stack(dropna=True).reset_index(drop=False)

all_columns_neurons = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_peaks_named_neurons.items()]
# all_columns_neurons = [my_reshape(d).drop(columns=['level_0']).rename(columns={0: k, 'level_1': 'dataset_idx'}) for k, d in all_peaks_named_neurons.items()]

from functools import reduce
df_neurons = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_neurons)
# df_neurons
# df_neurons = pd.concat(all_columns_neurons, join='outer', axis=1)
# df_neurons = df_neurons.loc[:,~df_neurons.columns.duplicated()].copy()


# Make sure the neurons are actually aligned with the other events


# df_neurons = pd.DataFrame.from_dict(all_peaks_named_neurons).T.explode(names_to_check).reset_index(drop=False)
# df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=True)
# df_curvature = pd.DataFrame.from_dict(all_curvature_peaks, orient='index').T.stack().reset_index(drop=True)

df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=False).rename(columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})
# df_curvature = pd.DataFrame.from_dict(all_curvature_peaks, orient='index').T.stack().reset_index(drop=False).rename(columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})

all_columns_curvature = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_curvature_peaks_dict.items()]
df_curvature = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_curvature)

all_dfs = [df_neurons, df_collision, df_curvature]
df_multi_neurons = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_dfs)

# df_multi_neurons = pd.concat([df_neurons, df_collision.astype(bool), df_curvature], join='inner', axis=1)
df_multi_neurons.columns = ('dataset_idx', 'dataset_name') + tuple(names_to_check) + ('collision_flag', ) + tuple(curvature_aliases)
df_multi_neurons['collision_flag'] = df_multi_neurons['collision_flag'].astype(bool)

df_multi_neurons['ventral_body_curvature'] = df_multi_neurons['summed_signed_curvature'] > 0


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['ventral_only_head_curvature']>0], x='ventral_only_head_curvature', y=name, trendline='ols',
#                     title='Correlation between curvature and post-reversal peak of ventral-only curvature')
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons,#[df_multi_neurons['summed_signed_curvature']>0], 
#                      x='summed_signed_curvature', y=name, trendline='ols', color='ventral_body_curvature',
#                     title="Correlation of neurons to peak in full-body curvature")
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['ventral_only_head_curvature']>0], x='ventral_only_curvature', y=name, trendline='ols', color='ventral_body_curvature',
#                     title='Correlation between curvature and post-reversal peak of ventral-only curvature')
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['summed_signed_curvature']>0], x='ventral_quantile_curvature', y=name, trendline='ols')
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['summed_signed_curvature']>0], x='ventral_only_curvature', y=name, trendline='ols')
#     fig.show()



# fig = px.scatter(df_multi_neurons, x='RIVL', y='RIVR', trendline='ols', title="RIVs are strongly correlated")
# fig.show()



fig = px.scatter(df_multi_neurons, x='RIVL', y='RID', trendline='ols',)
fig = add_trendline_annotation(fig)

apply_figure_settings(fig, width_factor=0.75, height_factor=0.3, plotly_not_matplotlib=True)
fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_RIV_correlation.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


fig = px.box(df_multi_neurons.groupby('dataset_name')[['RID', 'RIVL']].corr().unstack().iloc[:, 1].values, points='all')
fig.update_yaxes(title="Correlation per dataset", range=[-1, 1])
fig.update_xaxes(title="", showticklabels=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_RIV_correlation_boxplot.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)



fig = px.scatter(df_multi_neurons, x='RIVL', y='ventral_only_head_curvature', trendline='ols')
fig = add_trendline_annotation(fig)
fig.update_yaxes(title="Ventral only head curvature")
apply_figure_settings(fig, width_factor=0.75, height_factor=0.3, plotly_not_matplotlib=True)
fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_ventral_head_correlation.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


fig = px.box(df_multi_neurons.groupby('dataset_name')[['RID', 'ventral_only_head_curvature']].corr().unstack().iloc[:, 1].values, points='all')
fig.update_yaxes(title="Correlation per dataset", range=[-1, 1])
fig.update_xaxes(title="", showticklabels=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_ventral_head_correlation_boxplot.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# # Quantify the delay of RIV and RID peaks

from collections import defaultdict
# opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean')

names_to_check = [#'SMDDR', 'SMDDL', 
                  'SMDVL', 'SMDVR', 'RIVL', 'RIVR', 'RID']

all_peaks_timing_named_neurons = defaultdict(dict)
all_peaks_named_neurons = defaultdict(dict)
all_collision_flags = {}

curvature_aliases = ['ventral_only_head_curvature', 'summed_signed_curvature', 'ventral_quantile_curvature', 'head_signed_curvature', 'ventral_only_body_curvature']
all_curvature_peaks_dict = {a: {} for a in curvature_aliases}
for name, p in tqdm(all_projects_gcamp.items()):
    try:
        df_traces = p.calc_paper_traces()
        y_neurons = {n: df_traces[n] for n in names_to_check if n in df_traces}
                
        worm = p.worm_posture_class

        beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)
        
        all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)[0]
        
        for alias in all_curvature_peaks_dict.keys():
            y_curvature = worm.calc_behavior_from_alias(alias)
            curv_peaks, peak_times, rev_ends = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)
            all_curvature_peaks_dict[alias][name] = np.array(peak_times) - np.array(rev_ends)

        # Even if the neurons aren't found, the lists must be populated to be the same length as all others
        for _name in names_to_check:
            y = y_neurons.get(_name, None)
            if y is None:
                all_peaks_named_neurons[_name][name] = [np.nan]*len(curv_peaks)
                all_peaks_timing_named_neurons[_name][name] = [np.nan]*len(curv_peaks)
            else:
                curv_peaks, peak_times, rev_ends = worm.get_peaks_post_reversal(y, allow_reversal_before_peak=True)
                all_peaks_timing_named_neurons[_name][name] = np.array(peak_times) - np.array(rev_ends)
                all_peaks_named_neurons[_name][name] = curv_peaks

    except (ValueError, KeyError) as e:
        print(f"Error on dataset: {name}")
        print(e)
        continue


from functools import reduce
my_reshape = lambda d: pd.DataFrame.from_dict(d, orient='index').T.stack(dropna=True).reset_index(drop=False)

# Amplitude
all_columns_neurons = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_peaks_named_neurons.items()]
df_neurons = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_neurons)

# Timing
all_columns_neurons = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_peaks_timing_named_neurons.items()]
df_neurons_timing = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_neurons)


# Make sure the neurons are actually aligned with the other events


# df_neurons = pd.DataFrame.from_dict(all_peaks_named_neurons).T.explode(names_to_check).reset_index(drop=False)
# df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=True)
# df_curvature = pd.DataFrame.from_dict(all_curvature_peaks, orient='index').T.stack().reset_index(drop=True)

df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=False).rename(columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})
# df_curvature = pd.DataFrame.from_dict(all_curvature_peaks, orient='index').T.stack().reset_index(drop=False).rename(columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})

all_columns_curvature = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_curvature_peaks_dict.items()]
df_curvature = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_curvature)

all_dfs = [df_neurons, df_neurons_timing, df_collision, df_curvature]
df_multi_neurons_timing = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_dfs)

# df_multi_neurons = pd.concat([df_neurons, df_collision.astype(bool), df_curvature], join='inner', axis=1)
names_to_check_timing = [f"{n}_timing" for n in names_to_check]
df_multi_neurons_timing.columns = ('dataset_idx', 'dataset_name') + tuple(names_to_check) + tuple(names_to_check_timing) + ('collision_flag', ) + tuple(curvature_aliases)
df_multi_neurons_timing['collision_flag'] = df_multi_neurons['collision_flag'].astype(bool)

df_multi_neurons_timing['ventral_body_curvature'] = df_multi_neurons['summed_signed_curvature'] > 0


df_multi_neurons_timing.head()


# fig = px.box(df_multi_neurons_timing[['RID_timing', 'RIVL_timing', 'summed_signed_curvature']])
# from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
# fig = add_p_value_annotation(fig, array_columns=[[0,1], [1,2]])
# fig.show()


_df = df_multi_neurons_timing
# Remove the edges (clipping values)
df_timing_subset = _df[(13 > _df['RID_timing']) & (_df['RID_timing'] > 0.1) & (13 > _df['RIVL_timing']) & (_df['RIVL_timing'] > 0.1)]

fig = px.scatter(df_timing_subset, x='RID_timing', y='RIVL_timing', marginal_x='histogram', marginal_y='histogram', height=1000, width=1000,
                 # trendline='ols',
                )#title="Timing of RIV (RID) peaks after reversals are early (late)")
fig.update_layout(xaxis_title="RID Delay (seconds)", yaxis_title="RIV Delay (seconds)", font=dict(size=18))

apply_figure_settings(fig, width_factor=0.75, height_factor=0.3, plotly_not_matplotlib=True)
fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_RIV_delay_scatter.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


df_multi_neurons_timing['RID_timing'] - df_multi_neurons_timing['RIVL_timing']


df_multi_neurons_timing['diff'] = df_multi_neurons_timing['RID_timing'] - df_multi_neurons_timing['RIVL_timing']

fig = px.box(df_multi_neurons_timing.groupby('dataset_name').mean(), y='diff', points='all')
fig.update_yaxes(title="Mean delay per dataset (seconds)")
fig.update_xaxes(title="", showticklabels=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("postreversal_peaks", "RID_RIV_delay_boxplot.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# fig = px.scatter(df_multi_neurons_timing, x='RID_timing', y='RIVL_timing', color='RID',
#                  marginal_x='histogram', marginal_y='histogram', 
#                  height=1000, width=1000)
# fig.update_layout(xaxis_title="RID Delay (volumes)", yaxis_title="RIV Delay (volumes)", font=dict(size=18))
# fig.show()


# df_subset = df_multi_neurons_timing[df_multi_neurons_timing['RID_timing'] > 0]
# df_subset = df_subset[df_subset['summed_signed_curvature'] > 0]

# fig = px.scatter(df_subset, x='RID_timing', y='summed_signed_curvature',
#                  marginal_x='box', marginal_y='box', height=1000, width=1000,
#                  # trendline='ols',
#                 title="Timing of RID peaks against peaks of other time series")
# # fig.update_layout(xaxis_title="RID Delay (volumes)", yaxis_title="RIV Delay (volumes)", font=dict(size=18))
# fig.show()


# # df_subset = df_multi_neurons_timing[df_multi_neurons_timing['RID_timing'] > 0]
# # df_subset = df_subset[df_subset['summed_signed_curvature'] > 0]

# fig = px.scatter(df_multi_neurons_timing, x='RIVL_timing', y='summed_signed_curvature',
#                  marginal_x='box', marginal_y='box', height=1000, width=1000,
#                  # trendline='ols',
#                 title="Timing of RIVL peaks against peaks of other time series")
# # fig.update_layout(xaxis_title="RID Delay (volumes)", yaxis_title="RIV Delay (volumes)", font=dict(size=18))
# fig.show()


# px.scatter_matrix(df_multi_neurons_timing, dimensions=['RIVL_timing', 'RID_timing', 'summed_signed_curvature'], height=1000, width=1000)


# # Only plot the points that have 'real' peaks

# real_peak_ind = (0 < df_multi_neurons_timing['RID_timing']) & (49 > df_multi_neurons_timing['RID_timing']) & (0 < df_multi_neurons_timing['RIVL_timing']) & (49 > df_multi_neurons_timing['RIVL_timing'])

# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons_timing,#[real_peak_ind], 
#                      x='summed_signed_curvature', y=name, trendline='ols', color='ventral_body_curvature',
#                     title="Correlation of neurons to peak in full-body curvature")
#     fig.show()





# # Same but for SMD and RIV (as many as found), but the residuals

from collections import defaultdict
opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True, filter_mode='rolling_mean', residual_mode='pca')

names_to_check = ['RIVL', 'RIVR', 'RID']

all_peaks_named_neurons = defaultdict(dict)
all_collision_flags = {}

curvature_aliases = ['ventral_only_head_curvature', 'summed_signed_curvature', 'ventral_quantile_curvature', 'head_signed_curvature', 'ventral_only_curvature']
all_curvature_peaks_dict = {a: {} for a in curvature_aliases}
# all_curvature_peaks = {}
for name, p in tqdm(all_projects_gcamp.items()):
    try:
        df_traces = p.calc_default_traces(**opt, rename_neurons_using_manual_ids=True, manual_id_confidence_threshold=0)
        y_neurons = {n: df_traces[n] for n in names_to_check if n in df_traces}
                
        worm = p.worm_posture_class

        beh_annotation = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_collision = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.SELF_COLLISION).astype(int)
        all_collision_flags[name] = worm.get_peaks_post_reversal(y_collision, allow_reversal_before_peak=True)[0]
        
        for alias in all_curvature_peaks_dict.keys():
            y_curvature = worm.calc_behavior_from_alias(alias)
            curv_peaks = worm.get_peaks_post_reversal(y_curvature, allow_reversal_before_peak=True, use_idx_of_absolute_max=True)[0]
            all_curvature_peaks_dict[alias][name] = curv_peaks

        # Even if the neurons aren't found, the lists must be populated to be the same length as all others
        for _name in names_to_check:
            y = y_neurons.get(_name, None)
            if y is None:
                all_peaks_named_neurons[_name][name] = [np.nan]*len(curv_peaks)
            else:
                all_peaks_named_neurons[_name][name] = worm.get_peaks_post_reversal(y, allow_reversal_before_peak=True)[0]

    except (ValueError, KeyError) as e:
        print(f"Error on dataset: {name}")
        print(e)
        continue


my_reshape = lambda d: pd.DataFrame.from_dict(d, orient='index').T.stack(dropna=True).reset_index(drop=False)
all_columns_neurons = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_peaks_named_neurons.items()]

# from functools import reduce
df_neurons = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_neurons)
df_neurons


# Make sure the neurons are actually aligned with the other events

df_collision = pd.DataFrame.from_dict(all_collision_flags, orient='index').T.stack().reset_index(drop=False).rename(columns={'level_0': 'dataset_idx', 'level_1': 'dataset_name'})
all_columns_curvature = [my_reshape(d).rename(columns={0: k, 'level_0': 'dataset_idx', 'level_1': 'dataset_name'}) for k, d in all_curvature_peaks_dict.items()]
df_curvature = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_columns_curvature)

all_dfs = [df_neurons, df_collision, df_curvature]
df_multi_neurons_res = reduce(lambda left, right: pd.merge(left, right, on=['dataset_idx', 'dataset_name'], how='outer'), all_dfs)

df_multi_neurons_res.columns = ('dataset_idx', 'dataset_name') + tuple(names_to_check) + ('collision_flag', ) + tuple(curvature_aliases)
df_multi_neurons_res['collision_flag'] = df_multi_neurons_res['collision_flag'].astype(bool)

df_multi_neurons_res['ventral_body_curvature'] = df_multi_neurons_res['summed_signed_curvature'] > 0








for name in names_to_check:
    fig = px.scatter(df_multi_neurons_res[df_multi_neurons['ventral_only_head_curvature']>0], x='ventral_only_head_curvature', y=name, trendline='ols',
                    title='Correlation between curvature and post-reversal peak of ventral-only curvature')
    fig.show()


for name in names_to_check:
    fig = px.scatter(df_multi_neurons_res,#[df_multi_neurons['summed_signed_curvature']>0], 
                     x='summed_signed_curvature', y=name, trendline='ols', color='ventral_body_curvature',
                    title="Correlation of neurons to peak in full-body curvature")
    fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['ventral_only_head_curvature']>0], x='ventral_only_curvature', y=name, trendline='ols', color='ventral_body_curvature',
#                     title='Correlation between curvature and post-reversal peak of ventral-only curvature')
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['summed_signed_curvature']>0], x='ventral_quantile_curvature', y=name, trendline='ols')
#     fig.show()


# for name in names_to_check:
#     fig = px.scatter(df_multi_neurons[df_multi_neurons['summed_signed_curvature']>0], x='ventral_only_curvature', y=name, trendline='ols')
#     fig.show()



fig = px.scatter(df_multi_neurons_res, x='RIVL', y='RIVR', trendline='ols', title="RIVs are strongly correlated")
fig.show()



fig = px.scatter(df_multi_neurons_res, x='RIVL', y='RID', trendline='ols', title="RIV and RID are strongly correlated")
fig.show()























# # NOT DONE: RIS correlation to speed

worm = project_data_gcamp.worm_posture_class
df_traces = project_data_gcamp.calc_default_traces()

y_ris = df_traces['neuron_008']
y_speed = worm.worm_speed(fluorescence_fps=True, signed=True)
y_rev = worm.calc_behavior_from_alias('rev')


df = pd.DataFrame({'ris': y_ris, 'speed': y_speed, 'rev': y_rev})

px.scatter(df, x='ris', y='speed', color='rev', trendline='ols')


px.line(y_speed)


# ## Not done: Trigger to peaks of RIS instead of direct correlation

from scipy.signal import find_peaks


df_traces_filt = project_data_gcamp.calc_default_traces(filter_mode='rolling_mean')

x = df_traces_filt['neuron_008']
plt.figure(dpi=200)

project_data_gcamp.shade_axis_using_behavior()

peaks, properties = find_peaks(x, prominence=0.03)
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.show()


plt.hist(properties['prominences'], bins=100);


np.median(properties['prominences'])




















# # Scratch

from wbfm.utils.general.utils_behavior_annotation import shade_using_behavior_plotly


# Look at reannotating turns based on a more head-centered definition
worm = project_data_gcamp.worm_posture_class

beh = worm.beh_annotation(fluorescence_fps=True, reset_index=True)

y_ventral_body = worm.calc_behavior_from_alias('ventral_only_curvature')
y_dorsal_body = worm.calc_behavior_from_alias('dorsal_only_curvature')

y_ventral_head = worm.calc_behavior_from_alias('ventral_only_head_curvature')
y_dorsal_head = worm.calc_behavior_from_alias('dorsal_only_head_curvature')

y_ventral_interp = worm.calc_behavior_from_alias('interpolated_ventral_head_curvature')
y_dorsal_interp = worm.calc_behavior_from_alias('interpolated_dorsal_head_curvature')

df_amp = worm.hilbert_amplitude(fluorescence_fps=True, reset_index=True)
y_amp = df_amp.iloc[:, 1:5].mean(axis=1)

df = pd.DataFrame({'y_ventral_body': y_ventral_body, 'y_dorsal_body': y_dorsal_body, 
                   'y_ventral_head': y_ventral_head, 'y_dorsal_head': y_dorsal_head,
                   #'y_ventral_interp': y_ventral_interp, 'y_dorsal_interp': y_dorsal_interp,
#                  'y_amp': y_amp
})


fig = px.line(df)
shade_using_behavior_plotly(beh, fig)
fig.show()


plt.figure(figsize=(20,3))
plt.plot(df['y_ventral_head'])
plt.plot(df['y_dorsal_head'])
project_data_gcamp.shade_axis_using_behavior(additional_shaded_states=[BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN], DEBUG=True)


px.imshow(df_amp.T, aspect=5, zmin=0, zmax=0.1)


df_kymo = worm.curvature(fluorescence_fps=True, reset_index=True)
px.imshow(df_kymo.T, aspect=5, zmin=-0.05, zmax=0.05)


from scipy.signal._peak_finding import find_peaks


dat = y_ventral_head
height = np.mean(dat)
width = 2

peaks, properties = find_peaks(dat, height=height, width=width)
y_peaks = dat[peaks]


px.histogram(y_peaks)


# # Scratch
