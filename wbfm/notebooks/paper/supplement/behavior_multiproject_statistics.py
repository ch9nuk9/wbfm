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
import surpyval
import plotly.express as px


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# Load multiple datasets
from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets('gcamp')


all_projects_gfp = load_paper_datasets('gfp')


output_folder = "multiproject_behavior_quantifications"


# # Example dataset with zoom in

from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True);

to_save = True
if to_save:
    fname = os.path.join("behavior", "kymograph_with_time_series.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# ## Trajectory

xy = project_data_gcamp.worm_posture_class.stage_position(fluorescence_fps=False).copy()
xy = xy - xy.iloc[0, :]


import plotly.graph_objects as go

fig = px.scatter(xy, x='X', y='Y')
fig.update_yaxes(dict(title="Distance (mm)"))
fig.update_xaxes(dict(title="Distance (mm)"))

fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(
                    color='green',
                    size=20
                ), name='start'))

fig.add_trace(go.Scatter(x=[xy.iloc[-1, 0]], y=[xy.iloc[-1, 1]], marker=dict(
                    color='red',
                    size=20), name='end'
                ))
apply_figure_settings(fig, width_factor=0.4, height_factor=0.25, plotly_not_matplotlib=True)

fig.show()

    
fname = f"trajectory.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# ## Behavior transition diagram

dot, df_probabilities, df_raw_number = project_data_gcamp.worm_posture_class.plot_behavior_transition_diagram(output_folder='behavior')


# # Histograms of multiproject statistics

# ## Displacement

from collections import defaultdict

def calc_net_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()
    xy0 = df.loc[0, :][i_seg]
    xy1 = df.iloc[-1, :][i_seg]
    
    return np.linalg.norm(xy0 - xy1)
    
def calc_cumulative_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()[i_seg]
    dist = np.sqrt((df['X'] - df['X'].shift())**2 + (df['Y'] - df['Y'].shift())**2)
    line_integral = np.nansum(dist)
    
    return line_integral

def calc_displacement_dataframes(all_projects):
    
    all_displacements = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        all_displacements['net'][name] = calc_net_displacement(p)
        all_displacements['cumulative'][name] = calc_cumulative_displacement(p)
    df_displacement_gcamp = pd.DataFrame(all_displacements)
    
    return df_displacement_gcamp


df_displacement_gcamp = calc_displacement_dataframes(all_projects_gcamp)
df_displacement_gfp = calc_displacement_dataframes(all_projects_gfp)


df_displacement_gfp.shape


df_displacement_gcamp['genotype'] = 'gcamp'
df_displacement_gfp['genotype'] = 'gfp'

df_displacement = pd.concat([df_displacement_gcamp, df_displacement_gfp])


# fig = px.histogram(df_displacement, x='net', color='genotype', nbins=40)

# fig.update_layout(barmode='stack')
# # fig.update_traces(opacity=0.75)

# fig.show()


# fig = px.histogram(df_displacement, x='net', facet_row='genotype', color='genotype', nbins=30)
# fig.show()


# Alternative: boxplot with scatter plot
fig = px.box(df_displacement, y='net', x='genotype', color='genotype', points='all', title="Net displacement of animals in 8 minutes")
fig.update_layout(yaxis=dict(title='Distance (mm)'), xaxis=dict(title='Genotype'))

fig.update_layout(
    font=dict(size=24)
)
fig.show()

fname = "net_displacement.png"
fname = os.path.join(output_folder, fname)
fig.write_image(fname)

fig.write_image(fname.replace(".png", ".svg"))


# fig = px.histogram(df_displacement, x='cumulative', facet_row='genotype', color='genotype', nbins=30)
# fig.show()


# ## Speed, in several different ways

from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe
from wbfm.utils.general.utils_paper import apply_figure_settings


df_speed_gcamp = calc_speed_dataframe(all_projects_gcamp)
df_speed_gfp = calc_speed_dataframe(all_projects_gfp)


df_speed_gcamp['genotype'] = 'gcamp'
df_speed_gfp['genotype'] = 'gfp'

df_speed = pd.concat([df_speed_gcamp, df_speed_gfp])


# TODO: trace of hist (fewer bins), with error bars and gray with individual traces
# TODO: OR: normaly histogram, but boxplot per individual of rectified speeds


speed_types = [#'abs_stage_speed', 'middle_body_speed', 
               'signed_middle_body_speed']
for x in speed_types:
    fig = px.histogram(df_speed, x=x, facet_row='genotype', color='genotype', title="Speed",#x, 
                       histnorm='probability')
    fig.update_layout(title=dict(x=0.4, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(showlegend=True)
    fig.update_yaxes(dict(title="Probability", range=[0, 0.029]))
    fig.update_xaxes(dict(title="Speed (mm/s)", range=[-0.25, 0.19]))
    fig.update_xaxes(dict(title=""), row=2, col=1)
    
    apply_figure_settings(fig, width_factor=0.35, height_factor=0.25, plotly_not_matplotlib=True)
    
    fig.show()
    
    fname = f"{x}_histogram.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# fname = os.path.join(output_folder, "df_speed.h5")
# df_speed.to_hdf(fname, 'df_with_missing')


# ## Reversal and forward durations

from wbfm.utils.visualization.plot_summary_statistics import calc_durations_dataframe


df_duration_gcamp = calc_durations_dataframe(all_projects_gcamp)
df_duration_gfp = calc_durations_dataframe(all_projects_gfp)


# %debug


df_duration_gcamp['genotype'] = 'gcamp'
df_duration_gfp['genotype'] = 'gfp'

df_duration = pd.concat([df_duration_gcamp, df_duration_gfp])

fps = 3.5
df_duration['BehaviorCodes.FWD'] /= fps
df_duration['BehaviorCodes.REV'] /= fps


df_duration.columns



states = ['BehaviorCodes.FWD', 'BehaviorCodes.REV']
titles = ["forward", "reversal"]

for x, t in zip(states, titles):
    fig = px.histogram(df_duration, x=x, facet_row='genotype', color='genotype', title=f"Duration of {t} states", 
                       histnorm='probability')

    fig.update_layout(title=dict(x=0.5, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(xaxis_title="Time (s)", showlegend=False)
    if t == 'reversal':
        fig.update_xaxes(dict(range=[0, 20]))
        fig.update_yaxes(dict(range=[0, 0.19]))
    else:
        fig.update_xaxes(dict(range=[0, 90]))
        fig.update_yaxes(dict(range=[0, 0.5]))
                        
    apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
    fig.show()
    
    fname = f"duration_histogram_{x.split('.')[1]}.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# fname = os.path.join(output_folder, "df_durations.h5")
# df_duration.to_hdf(fname, 'df_with_missing')


# ## Reversal and forward frequency

from wbfm.utils.visualization.plot_summary_statistics import calc_onset_frequency_dataframe


df_frequency_gcamp = calc_onset_frequency_dataframe(all_projects_gcamp)
df_frequency_gfp = calc_onset_frequency_dataframe(all_projects_gfp)


df_frequency_gcamp['genotype'] = 'gcamp'
df_frequency_gfp['genotype'] = 'gfp'

df_frequency = pd.concat([df_frequency_gcamp, df_frequency_gfp])

# df_frequency['BehaviorCodes.FWD'] /= fps
# df_frequency['BehaviorCodes.REV'] /= fps



states = ['BehaviorCodes.FWD', 'BehaviorCodes.REV']
titles = ["forward", "reversal"]

for x, t in zip(states, titles):
    fig = px.histogram(df_frequency, x=x, facet_row='genotype', color='genotype', title=f"Frequency of {t} states", 
                       histnorm='probability')
    
    fig.update_layout(title=dict(x=0.5, y=0.99))
    fig.update_layout(
        xaxis_title="Frequency (1/min)", showlegend=False
    )
    fig.update_yaxes(dict(range=[0, 0.59]))
    apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
    
    for anno in fig['layout']['annotations']:
        anno['text']=''
    
    fig.show()
    
    fname = f"frequency_histogram_{x.split('.')[1]}.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# ## Other states: SLOWING

# ### Frequency

from wbfm.utils.visualization.plot_summary_statistics import calc_onset_frequency_dataframe
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


df_frequency_gcamp = calc_onset_frequency_dataframe(all_projects_gcamp, states=[BehaviorCodes.SLOWING])
df_frequency_gfp = calc_onset_frequency_dataframe(all_projects_gfp, states=[BehaviorCodes.SLOWING])


df_frequency_gcamp['genotype'] = 'gcamp'
df_frequency_gfp['genotype'] = 'gfp'

df_frequency = pd.concat([df_frequency_gcamp, df_frequency_gfp])


states = ['BehaviorCodes.SLOWING']
titles = ["slowing"]

for x, t in zip(states, titles):
    fig = px.histogram(df_frequency, x=x, facet_row='genotype', color='genotype', title=f"Frequency of {t} states", histnorm='probability')
    
    fig.update_layout(
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis_title="Frequency (1/min)"
    )
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.show()
    
    fname = f"frequency_histogram_SLOWING_{x.split('.')[1]}.png"
    fname = os.path.join(output_folder, fname)
    fig.write_image(fname)
    fig.write_image(fname.replace(".png", ".svg"))


# ### Duration

from wbfm.utils.visualization.plot_summary_statistics import calc_durations_dataframe


df_duration_gcamp = calc_durations_dataframe(all_projects_gcamp, states=[BehaviorCodes.SLOWING])
df_duration_gfp = calc_durations_dataframe(all_projects_gfp, states=[BehaviorCodes.SLOWING])


df_duration_gcamp['genotype'] = 'gcamp'
df_duration_gfp['genotype'] = 'gfp'

df_duration = pd.concat([df_duration_gcamp, df_duration_gfp])

fps = 3.5
df_duration['BehaviorCodes.SLOWING'] /= fps



states = ['BehaviorCodes.SLOWING']
titles = ["slowing"]

for x, t in zip(states, titles):
    fig = px.histogram(df_duration, x=x, facet_row='genotype', color='genotype', title=f"Duration of {t} states", histnorm='probability')
    
    fig.update_layout(
        font=dict(size=24)
    )
    fig.update_layout(
        xaxis_title="Time (s)"
    )
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.show()
    
    fname = f"duration_histogram_SLOWING_{x.split('.')[1]}.png"
    fname = os.path.join(output_folder, fname)
    fig.write_image(fname)
    fig.write_image(fname.replace(".png", ".svg"))








# # Histogram of post-reversal head bend peaks

from wbfm.utils.general.utils_paper import apply_figure_settings


# For each project, get the positive and negative post reversal peaks
# Use the summed signed head curvature

final_ventral_dict = {}
final_dorsal_dict = {}

for name, p in tqdm(all_projects_gcamp.items()):

    worm = p.worm_posture_class
    y_curvature = worm.calc_behavior_from_alias('head_signed_curvature')

    ventral_peaks, ventral_peak_times, _ = worm.get_peaks_post_reversal(y_curvature, num_points_after_reversal=20)
    dorsal_peaks, dorsal_peak_times, all_rev_ends = worm.get_peaks_post_reversal(-y_curvature, num_points_after_reversal=20)

    # Keep the positive peak if it is a ventral turn at that time, or negative peak if it is dorsal
    # ... actually I don't think Ulises' annotations are that frame-accurate, so I will take whichever peak is closer to the end of the reversal, i.e. the first body bend
    ventral_to_keep = []
    dorsal_to_keep = []
    for vp, vt, dp, dt, end in zip(ventral_peaks, ventral_peak_times, dorsal_peaks, dorsal_peak_times, all_rev_ends):
        # Alternate: take the one with the higher amplitude
        # if np.abs(vp) > np.abs(dp):
        #     ventral_to_keep.append(vp)
        # else:
        #     dorsal_to_keep.append(dp)
        
        # Skip if the event is exactly at the end
#         vt_at_edge = vt == end
#         vt_early = vt < dt
#         dt_at_edge = dt == end
        
        if vt < dt and vt > end:
            ventral_to_keep.append(vp)
        elif dt > end:
            dorsal_to_keep.append(-dp)
        else:
            pass
    final_ventral_dict[name] = ventral_to_keep
    final_dorsal_dict[name] = dorsal_to_keep


# For now, ignore the dataset they came from
df_ventral = pd.DataFrame(np.concatenate(list(final_ventral_dict.values())))
df_ventral['Turn Direction'] = 'Ventral'
df_dorsal = pd.DataFrame(np.concatenate(list(final_dorsal_dict.values())))
df_dorsal['Turn Direction'] = 'Dorsal'

df_turns = pd.concat([df_ventral, df_dorsal])
df_turns.columns = ['Amplitude', 'Turn Direction']


beh_list = [BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN]
cmap = [BehaviorCodes.ethogram_cmap()[beh] for beh in beh_list]


df_turns['Amplitude'] = df_turns['Amplitude'].abs()

fig = px.histogram(df_turns, color="Turn Direction", histnorm='probability', color_discrete_sequence=cmap,
                  barmode='overlay')
fig.update_layout(xaxis=dict(title="Peak Head Curvature"), showlegend=False)
apply_figure_settings(fig, width_factor=0.35, height_factor=0.15, plotly_not_matplotlib=True)

fig.show()

fname = f"first_head_bend_absolute_curvature_histogram.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


fig = px.pie(df_turns, names="Turn Direction", color_discrete_sequence=cmap)

apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.show()

fname = f"first_head_bend_absolute_curvature_pie_chart.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# fname = os.path.join(output_folder, "df_dorsal_ventral.h5")
# df_turns.to_hdf(fname, 'df_with_missing')





# # Histograms of new ventral annotations

from wbfm.utils.external.utils_pandas import get_dataframe_of_transitions
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


beh_vec = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
beh_vec = [b.value for b in beh_vec]
df_transition = get_dataframe_of_transitions(beh_vec, convert_to_probabilities=True)


mapper = lambda val: BehaviorCodes(val).name

df_transition = df_transition#.rename(columns=mapper).rename(index=mapper)
df_transition


# For each project, get the transition probability dataframe
all_transitions = []
for name, p in tqdm(all_projects_gcamp.items()):
    beh_vec = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
    beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
    beh_vec = [b.value for b in beh_vec]
    df_transition = get_dataframe_of_transitions(beh_vec, convert_to_probabilities=False, ignore_diagonal=True)
    
    all_transitions.append(df_transition)
df_all_transitions = sum(all_transitions)


mapper = lambda val: BehaviorCodes(val).name

df = df_all_transitions.rename(columns=mapper).rename(index=mapper)
# df = df.div(df.sum(axis=1), axis=0)
df.index.name = None
df.columns.name = None
df


px.bar(df.loc['REV', :])


import plotly.graph_objs as go

beh_list = [BehaviorCodes.VENTRAL_TURN, BehaviorCodes.FWD, BehaviorCodes.DORSAL_TURN]
cmap = [BehaviorCodes.ethogram_cmap()[beh] for beh in beh_list]

df_subset = df.loc['REV', :]
df_subset = df_subset[df_subset > 0].reset_index()
df_subset.loc[0, 'index'] = 'AMBIGUOUS'  # Rename the FWD transition

fig = px.pie(df_subset.reset_index(), values='REV', names='index', color_discrete_sequence=cmap)

apply_figure_settings(fig, width_factor=0.4, height_factor=0.25, plotly_not_matplotlib=True)
fig.show()
# fig = go.Figure(data=[go.Pie(labels=df_subset.index, values=df_subset)])

fname = f"first_head_bend_absolute_curvature_pie_chart.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# # Summary plots, to be saved in their own folders




from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior
from wbfm.utils.projects.finished_project_data import load_all_projects_in_folder


# Save these for all the datasets that Ulises annotated head casts
folder = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar'

all_projects = load_all_projects_in_folder(folder)


# df = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
# df.value_counts()


# fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True)


# %debug


for name, p in tqdm(all_projects.items()):
    try:
        fig = make_summary_interactive_kymograph_with_behavior(p, to_save=True, to_show=False)
        print(name)
    except:
        pass


# # Calculate the cumulative distribution of forward durations, including censoring information

duration_vec = []
censored_vec = []
for name, p in all_projects_gcamp.items():
    ind_class = p.worm_posture_class.calc_triggered_average_indices(ind_preceding=0, min_duration=0)
    d, c = ind_class.all_durations_with_censoring()
    duration_vec.extend(d)
    censored_vec.extend(c)


# With censoring
model = surpyval.Weibull.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# Package everything for saving
import pickle
y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

out = dict(duration_vec=duration_vec, censored_vec=censored_vec, 
           x_dat=x_dat, y_dat=y_dat,
           alpha=model.alpha, beta=model.beta, model_dict=model.to_dict())

foldername = "/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior"
fname = os.path.join(foldername, 'forward_duration.pickle')

with open(fname, 'wb') as f:
    pickle.dump(out, f)


x = np.arange(1000)
y = model.ff(x)

y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

plt.plot(y)
plt.plot(x_dat, y_dat)


# ## Same but for reversal distribution

from wbfm.utils.external.utils_behavior_annotation import BehaviorCodes


duration_vec = []
censored_vec = []
for name, p in all_projects_gcamp.items():
    ind_class = p.worm_posture_class.calc_triggered_average_indices(ind_preceding=0, min_duration=0, state=BehaviorCodes.REV)
    d, c = ind_class.all_durations_with_censoring()
    duration_vec.extend(d)
    censored_vec.extend(c)


# With censoring
model = surpyval.Weibull.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# Package everything for saving
import pickle
y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

out = dict(duration_vec=duration_vec, censored_vec=censored_vec, 
           x_dat=x_dat, y_dat=y_dat,
           alpha=model.alpha, beta=model.beta, model_dict=model.to_dict())

foldername = "/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior"
fname = os.path.join(foldername, 'reversal_duration.pickle')

with open(fname, 'wb') as f:
    pickle.dump(out, f)


x = np.arange(1000)
y = model.ff(x)

y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

plt.plot(y)
plt.plot(x_dat, y_dat)


# With censoring
model = surpyval.Exponential.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()











# # Scratch: other distributions


model = surpyval.Exponential.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# NO CENSORING
import surpyval

model = surpyval.Weibull.fit(x=duration_vec)
print(model)
print(model.aic())
model.plot()


# See: https://surpyval.readthedocs.io/en/latest/applications.html
from autograd import numpy as np_auto

bounds = ((0, None), (0, None), (0, None),)
support = (0, np_auto.inf)
param_names = ['lambda', 'alpha', 'beta']
def Hf(x, *params):
    Hf = params[0] * x + (params[1]/params[2])*(np_auto.exp(params[2]*x))
    return Hf
GompertzMakeham = surpyval.Distribution('GompertzMakeham', Hf, param_names, bounds, support)

model = GompertzMakeham.fit(x=duration_vec / np.max(duration_vec), c=censored_vec, how='MLE')
model.plot(alpha_ci=0.95)
print(model)
print(model.aic())


# See: https://surpyval.readthedocs.io/en/latest/applications.html
from autograd import numpy as np_auto

bounds = ((0, None), (0, None), (0, None), (0, None))
support = (0, np_auto.inf)
param_names = ['alpha1', 'lambda1', 'alpha2', 'lambda2']
def Hf(x, *params):
    Hf = params[0]*(np_auto.exp(params[1]*x)) + params[2]*(np_auto.exp(params[3]*x))
    return Hf
DoubleExponential = surpyval.Distribution('double_exponential', Hf, param_names, bounds, support)

# model = DoubleExponential.fit(x=duration_vec, c=censored_vec, init=initial)
initial = (0.5, 0.01, 0.5, 0.5)
model = DoubleExponential.fit(x=duration_vec / np.max(duration_vec), c=censored_vec, init=initial, how='MLE')
model.plot(alpha_ci=0.95, heuristic='Nelson-Aalen')
print(model)


y = surpyval.Weibull.random(1000, 34, 0.63)
n = plt.hist(y, cumulative=True, bins=1000, log=True);




