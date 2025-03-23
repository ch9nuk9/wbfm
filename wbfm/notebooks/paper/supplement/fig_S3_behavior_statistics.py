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
import surpyval
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
import plotly.express as px


# In[3]:


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets('gcamp')


# In[5]:


all_projects_gfp = load_paper_datasets('gfp')


# In[6]:


output_folder = "multiproject_behavior_quantifications"


# # Example dataset with zoom in

# In[7]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[8]:


from wbfm.utils.visualization.plot_traces import build_all_plot_variables_for_summary_plot
# Test: look at variables that I'm plotting
behavior_alias_dict = {'Turns': ['dorsal_turn', 'ventral_turn'],
                               'Other': ['self_collision', 'head_cast'],
                               'Rev': ['rev']}
additional_shaded_states = []

column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, _row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, var_explained_line, var_explained_line_opt, weights_list, weights_opt_list = build_all_plot_variables_for_summary_plot(
        project_data_gcamp, 3, use_behavior_traces=True, behavior_alias_dict=behavior_alias_dict,
        additional_shaded_states=additional_shaded_states, showlegend=False)


# In[9]:


fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True,
                                                      apply_figure_size_settings=True, showlegend=False,
                                                       row_heights=[0.25, 0.05, 0.2, 0.2, 0.2])

to_save = True
if to_save:
    fname = os.path.join("behavior", "kymograph_with_time_series.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# In[12]:


fps = project_data_gcamp.physical_unit_conversion.frames_per_second
fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True,
                                                      apply_figure_size_settings=True, discrete_behaviors=True,
                                                       row_heights=[0.25, 0.05, 0.2, 0.2, 0.2],
                                                      x_range=[31000/fps, 35000/fps])

to_save = True
if to_save:
    fname = os.path.join("behavior", "kymograph_with_discrete_time_series.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# In[13]:


fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True,
                                                      apply_figure_size_settings=True, eigenworm_behaviors=True,
                                                       row_heights=[0.25, 0.05, 0.2, 0.2, 0.2],
                                                      #x_range=[31000, 35000]
                                                      )

to_save = True
if to_save:
    fname = os.path.join("behavior", "kymograph_with_eigenworm_time_series.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# In[12]:


get_ipython().run_line_magic('debug', '')


# ## Trajectory

# In[14]:


from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.utils_paper import behavior_name_mapping


# In[15]:


xy = project_data_gcamp.worm_posture_class.calc_behavior_from_alias('worm_center_position').copy()
# xy = project_data_gcamp.worm_posture_class.stage_position(fluorescence_fps=True).copy()
xy = xy - xy.iloc[0, :]

beh = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True, simplify_states=True,
                                                           include_head_cast=False, include_collision=False, include_pause=True)

df_xy = xy
df_xy['Behavior'] = beh.values
# df_xy['Behavior'] = df_xy['Behavior'].map(lambda x: behavior_name_mapping()[x.name])
df_xy.head()


# In[16]:


# import plotly.graph_objects as go

# df_xy['size'] = 1

# fig = px.scatter(df_xy, x='X', y='Y')
# fig.update_yaxes(dict(title="Distance (mm)"))
# fig.update_xaxes(dict(title="Distance (mm)"))

# fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(
#                     color='green',
#                     size=5
#                 ), name='start'))

# fig.add_trace(go.Scatter(x=[xy.iloc[-1, 0]], y=[xy.iloc[-1, 1]], marker=dict(
#                     color='red',
#                     size=5), name='end'
#                 ))
# apply_figure_settings(fig, width_factor=0.5, height_factor=0.25, plotly_not_matplotlib=True)

# fig.update_traces(marker=dict(line=dict(width=0)))
# fig.show()

    
# fname = f"trajectory.png"
# fname = os.path.join("behavior", fname)
# fig.write_image(fname, scale=3)
# fig.write_image(fname.replace(".png", ".svg"))


# In[17]:


import plotly.graph_objects as go

df_xy['size'] = 1
ethogram_cmap = BehaviorCodes.ethogram_cmap(include_turns=True, include_reversal_turns=False)
df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_xy, ['X', 'Y'], 'Behavior')

# Loop to prep each line, then plot
state_codes = df_xy['Behavior'].unique()
phase_plot_list = []
for i, state_code in enumerate(state_codes):
    if state_code == BehaviorCodes.UNKNOWN:
        continue
    phase_plot_list.append(
                go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], mode='lines',
                             name=behavior_name_mapping()[state_code.full_name], line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

fig = go.Figure()
fig.add_traces(phase_plot_list)

# fig = px.scatter(df_xy, x='X', y='Y', color='Behavior')
fig.update_yaxes(dict(title="Distance (mm)"), scaleanchor= 'x')
fig.update_xaxes(dict(title="Distance (mm)"), )#range=[-1, 1])

fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(
                    color='black', symbol='x',
                    size=10
                ), name='start'))

fig.add_trace(go.Scatter(x=[xy.iloc[-1, 0]], y=[xy.iloc[-1, 1]], marker=dict(
                    color='black',
                    size=10), name='end'
                ))
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25, plotly_not_matplotlib=True)

fig.update_traces(marker=dict(line=dict(width=0)))
fig.show()

    
fname = f"trajectory.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# ## NOT USING: Behavior transition diagram

# In[18]:


# dot, df_probabilities, df_raw_number = project_data_gcamp.worm_posture_class.plot_behavior_transition_diagram(output_folder='behavior')


# # Histograms of multiproject statistics

# ## Displacement

# In[19]:


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


# In[20]:


df_displacement_gcamp = calc_displacement_dataframes(all_projects_gcamp)
df_displacement_gfp = calc_displacement_dataframes(all_projects_gfp)


# In[21]:


df_displacement_gfp.shape


# In[22]:


df_displacement_gcamp['genotype'] = 'gcamp'
df_displacement_gfp['genotype'] = 'gfp'

df_displacement = pd.concat([df_displacement_gcamp, df_displacement_gfp])


# In[23]:


# fig = px.histogram(df_displacement, x='net', color='genotype', nbins=40)

# fig.update_layout(barmode='stack')
# # fig.update_traces(opacity=0.75)

# fig.show()


# In[24]:


# fig = px.histogram(df_displacement, x='net', facet_row='genotype', color='genotype', nbins=30)
# fig.show()


# In[25]:


# Alternative: boxplot with scatter plot
fig = px.box(df_displacement, y='net', x='genotype', color='genotype', points='all', title="Net displacement of animals in 8 minutes")
fig.update_layout(yaxis=dict(title='Distance (mm)'), xaxis=dict(title='Genotype'))

fig.update_layout(
    font=dict(size=24)
)
fig.show()

fname = "net_displacement.png"
fname = os.path.join('behavior', fname)
fig.write_image(fname)

fig.write_image(fname.replace(".png", ".svg"))


# In[26]:


# fig = px.histogram(df_displacement, x='cumulative', facet_row='genotype', color='genotype', nbins=30)
# fig.show()


# ## Speed, in several different ways

# In[27]:


from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[28]:


df_speed_gcamp = calc_speed_dataframe(all_projects_gcamp)
df_speed_gfp = calc_speed_dataframe(all_projects_gfp)


# In[ ]:





# In[29]:


from wbfm.utils.general.utils_paper import data_type_name_mapping

df_speed_gcamp['Genotype'] = 'gcamp'
df_speed_gfp['Genotype'] = 'gfp'
df_speed = pd.concat([df_speed_gcamp, df_speed_gfp])

df_speed['Genotype'] = df_speed['Genotype'].map(data_type_name_mapping())
speed_types = [#'abs_stage_speed', 'middle_body_speed', 
               'signed_middle_body_speed']
for x in speed_types:
    fig = px.histogram(df_speed, x=x, facet_row='Genotype', color='Genotype', color_discrete_map=plotly_paper_color_discrete_map(), #title="Speed",#x, 
                       histnorm='probability')
    fig.update_layout(title=dict(x=0.4, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(showlegend=True)
    # fig.update_yaxes(dict(title="Probability", range=[0, 0.019]))
    fig.update_yaxes(dict(title="Probability", range=[0, 0.029]))
    fig.update_xaxes(dict(title="Velocity (mm/s)", range=[-0.25, 0.19]))
    fig.update_xaxes(dict(title=""), row=2, col=1)
    
    fig.update_traces(xbins=dict( # bins used for histogram
        size=0.002
    ))
    apply_figure_settings(fig, width_factor=0.4, height_factor=0.2, plotly_not_matplotlib=True)
    
    fig.show()
    
    fname = f"{x}_histogram.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# In[30]:


# fname = os.path.join(output_folder, "df_speed.h5")
# df_speed.to_hdf(fname, 'df_with_missing')


# ## Reversal and forward durations

# In[31]:


from wbfm.utils.visualization.plot_summary_statistics import calc_durations_dataframe


# In[32]:


df_duration_gcamp = calc_durations_dataframe(all_projects_gcamp)
df_duration_gfp = calc_durations_dataframe(all_projects_gfp)


# In[33]:


# %debug


# In[34]:


df_duration_gcamp['genotype'] = 'gcamp'
df_duration_gfp['genotype'] = 'gfp'

df_duration = pd.concat([df_duration_gcamp, df_duration_gfp])

fps = 3.5
df_duration['BehaviorCodes.FWD'] /= fps
df_duration['BehaviorCodes.REV'] /= fps


# In[35]:


df_duration.columns


# In[46]:



states = ['BehaviorCodes.FWD', 'BehaviorCodes.REV']
titles = ["Fwd", "Bwd"]

for x, t in zip(states, titles):
    fig = px.histogram(df_duration, x=x, facet_row='genotype', color='genotype', color_discrete_map=plotly_paper_color_discrete_map(), 
                       title=f"<br>                {t} duration", 
                       # title=f"                              {t} duration", 
                       histnorm='probability')

    fig.update_layout(title=dict(x=0.5, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(xaxis_title="Time (s)", showlegend=False)
    fig.update_traces(xbins=dict( # bins used for histogram
        # start=0.0,
        # end=60.0,
        size=1
    ))
    fig.update_xaxes(dict(range=[0, 90]))
    fig.update_yaxes(dict(range=[0, 0.19]), title="")
    width_factor = 0.2
    if t == 'Reversal':
        # fig.update_xaxes(dict(range=[0, 20]))
        fig.update_yaxes(showticklabels=False, overwrite=True)
        width_factor -= 0.01
    # else:
    #     fig.update_xaxes(dict(range=[0, 90]))
    #     fig.update_yaxes(dict(range=[0, 0.19]))
                        
    apply_figure_settings(fig, width_factor=width_factor, height_factor=0.2, plotly_not_matplotlib=True)
    fig.show()
    
    fname = f"duration_histogram_{x.split('.')[1]}.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# In[37]:



fig = px.histogram(df_duration.melt(id_vars=['dataset_name', 'genotype']), 
                   x='value', facet_row='genotype', color='genotype',
                   facet_col='variable',
                   color_discrete_map=plotly_paper_color_discrete_map(), 
                   title=f"<br>                {t} duration", 
                   histnorm='probability')


# In[ ]:





# In[37]:


# fname = os.path.join(output_folder, "df_durations.h5")
# df_duration.to_hdf(fname, 'df_with_missing')


# ## Reversal and forward frequency

# In[38]:


from wbfm.utils.visualization.plot_summary_statistics import calc_onset_frequency_dataframe


# In[39]:


df_frequency_gcamp = calc_onset_frequency_dataframe(all_projects_gcamp)
df_frequency_gfp = calc_onset_frequency_dataframe(all_projects_gfp)


# In[40]:


df_frequency_gcamp['genotype'] = 'gcamp'
df_frequency_gfp['genotype'] = 'gfp'

df_frequency = pd.concat([df_frequency_gcamp, df_frequency_gfp])

# df_frequency['BehaviorCodes.FWD'] /= fps
# df_frequency['BehaviorCodes.REV'] /= fps


# In[41]:



states = ['BehaviorCodes.FWD', 'BehaviorCodes.REV']
titles = ["Forward", "Reversal"]

for x, t in zip(states, titles):
    fig = px.histogram(df_frequency, x=x, facet_row='genotype', color='genotype', title=f"<br>     {t} frequency", 
                       histnorm='probability', color_discrete_map=plotly_paper_color_discrete_map())
    
    fig.update_layout(title=dict(x=0.5, y=0.99))
    fig.update_layout(
        xaxis_title="Frequency (1/min)", showlegend=False
    )
    fig.update_yaxes(dict(range=[0, 0.69]))
    fig.update_traces(xbins=dict( # bins used for histogram
        size=1
    ))
    apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
    
    for anno in fig['layout']['annotations']:
        anno['text']=''
    
    fig.show()
    
    fname = f"frequency_histogram_{x.split('.')[1]}.png"
    fname = os.path.join("behavior", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# # Histogram of post-reversal head bend peaks

# In[42]:


from wbfm.utils.general.utils_paper import apply_figure_settings


# In[43]:


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


# In[44]:


# For now, ignore the dataset they came from
df_ventral = pd.DataFrame(np.concatenate(list(final_ventral_dict.values())))
df_ventral['Turn Direction'] = 'Ventral'
df_dorsal = pd.DataFrame(np.concatenate(list(final_dorsal_dict.values())))
df_dorsal['Turn Direction'] = 'Dorsal'

df_turns = pd.concat([df_ventral, df_dorsal])
df_turns.columns = ['Amplitude', 'Turn Direction']
df_turns['Amplitude'] *= 1000  # Change to 1/mm


# In[45]:


beh_list = [BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN]
cmap = [BehaviorCodes.ethogram_cmap()[beh] for beh in beh_list]


# In[46]:


df_turns['Amplitude'] = df_turns['Amplitude'].abs()

fig = px.histogram(df_turns, color="Turn Direction", histnorm='probability', color_discrete_sequence=cmap,
                  barmode='overlay')
fig.update_layout(xaxis=dict(title="Peak Head Curvature (1/mm)"), showlegend=False)
fig.update_layout(yaxis=dict(title="Probability"), showlegend=False)
fig.update_traces(xbins=dict( # bins used for histogram
    # start=0.0,
    # end=60.0,
    size=1
))
fig.update_yaxes(range=[0, 0.25])

fig.update_layout(
    showlegend=True,
    legend=dict(
      yanchor="middle",
      y=1.75,
      xanchor="left",
      x=0.5
    )
)
apply_figure_settings(fig, width_factor=0.32, height_factor=0.14, plotly_not_matplotlib=True)

fig.show()

fname = f"first_head_bend_absolute_curvature_histogram.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[47]:


fig = px.pie(df_turns, names="Turn Direction", color_discrete_sequence=cmap)

apply_figure_settings(fig, width_factor=0.15, height_factor=0.1, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
fig.show()

fname = f"first_head_bend_absolute_curvature_pie_chart.png"
fname = os.path.join("behavior", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[48]:


# fname = os.path.join(output_folder, "df_dorsal_ventral.h5")
# df_turns.to_hdf(fname, 'df_with_missing')


# In[ ]:





# # Histograms of new ventral annotations

# In[49]:


from wbfm.utils.external.utils_pandas import get_dataframe_of_transitions
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[50]:


beh_vec = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
beh_vec = [b.value for b in beh_vec]
df_transition = get_dataframe_of_transitions(beh_vec, convert_to_probabilities=True)


# In[51]:


mapper = lambda val: BehaviorCodes(val).name

df_transition = df_transition#.rename(columns=mapper).rename(index=mapper)
df_transition


# In[52]:


# For each project, get the transition probability dataframe
all_transitions = []
for name, p in tqdm(all_projects_gcamp.items()):
    beh_vec = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
    beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
    beh_vec = [b.value for b in beh_vec]
    df_transition = get_dataframe_of_transitions(beh_vec, convert_to_probabilities=False, ignore_diagonal=True)
    
    all_transitions.append(df_transition)
df_all_transitions = sum(all_transitions)


# In[53]:


mapper = lambda val: BehaviorCodes(val).name

df = df_all_transitions.rename(columns=mapper).rename(index=mapper)
# df = df.div(df.sum(axis=1), axis=0)
df.index.name = None
df.columns.name = None
df


# In[54]:


px.bar(df.loc['REV', :])


# In[55]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SCRATCH

# ## Other states: SLOWING

# ### Frequency

# In[56]:


from wbfm.utils.visualization.plot_summary_statistics import calc_onset_frequency_dataframe
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[57]:


df_frequency_gcamp = calc_onset_frequency_dataframe(all_projects_gcamp, states=[BehaviorCodes.SLOWING])
df_frequency_gfp = calc_onset_frequency_dataframe(all_projects_gfp, states=[BehaviorCodes.SLOWING])


# In[58]:


df_frequency_gcamp['genotype'] = 'gcamp'
df_frequency_gfp['genotype'] = 'gfp'

df_frequency = pd.concat([df_frequency_gcamp, df_frequency_gfp])


# In[59]:


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

# In[ ]:


from wbfm.utils.visualization.plot_summary_statistics import calc_durations_dataframe


# In[ ]:


df_duration_gcamp = calc_durations_dataframe(all_projects_gcamp, states=[BehaviorCodes.SLOWING])
df_duration_gfp = calc_durations_dataframe(all_projects_gfp, states=[BehaviorCodes.SLOWING])


# In[ ]:


df_duration_gcamp['genotype'] = 'gcamp'
df_duration_gfp['genotype'] = 'gfp'

df_duration = pd.concat([df_duration_gcamp, df_duration_gfp])

fps = 3.5
df_duration['BehaviorCodes.SLOWING'] /= fps


# In[ ]:



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


# In[ ]:





# In[ ]:





# # Summary plots, to be saved in their own folders

# In[ ]:





# In[ ]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior
from wbfm.utils.projects.finished_project_data import load_all_projects_in_folder


# In[ ]:


# Save these for all the datasets that Ulises annotated head casts
folder = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar'

all_projects = load_all_projects_in_folder(folder)


# In[ ]:


# df = project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)
# df.value_counts()


# In[ ]:


# fig = make_summary_interactive_kymograph_with_behavior(project_data_gcamp, to_save=False, to_show=True)


# In[ ]:


# %debug


# In[ ]:


for name, p in tqdm(all_projects.items()):
    try:
        fig = make_summary_interactive_kymograph_with_behavior(p, to_save=True, to_show=False)
        print(name)
    except:
        pass


# # Calculate the cumulative distribution of forward durations, including censoring information

# In[ ]:


duration_vec = []
censored_vec = []
for name, p in all_projects_gcamp.items():
    ind_class = p.worm_posture_class.calc_triggered_average_indices(ind_preceding=0, min_duration=0)
    d, c = ind_class.all_durations_with_censoring()
    duration_vec.extend(d)
    censored_vec.extend(c)


# In[ ]:


# With censoring
model = surpyval.Weibull.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# In[ ]:


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


# In[ ]:


x = np.arange(1000)
y = model.ff(x)

y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

plt.plot(y)
plt.plot(x_dat, y_dat)


# ## Same but for reversal distribution

# In[ ]:


from wbfm.utils.external.utils_behavior_annotation import BehaviorCodes


# In[ ]:


duration_vec = []
censored_vec = []
for name, p in all_projects_gcamp.items():
    ind_class = p.worm_posture_class.calc_triggered_average_indices(ind_preceding=0, min_duration=0, state=BehaviorCodes.REV)
    d, c = ind_class.all_durations_with_censoring()
    duration_vec.extend(d)
    censored_vec.extend(c)


# In[ ]:


# With censoring
model = surpyval.Weibull.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# In[ ]:


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


# In[ ]:


x = np.arange(1000)
y = model.ff(x)

y_dat, x_dat = np.histogram(duration_vec, bins=np.arange(1000))
x_dat = x_dat[:-1]
y_dat = np.cumsum(y_dat / np.sum(y_dat))

plt.plot(y)
plt.plot(x_dat, y_dat)


# In[ ]:


# With censoring
model = surpyval.Exponential.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# # Scratch: other distributions

# In[ ]:



model = surpyval.Exponential.fit(x=duration_vec, c=censored_vec)
print(model)
print(model.aic())
model.plot()


# In[ ]:


# NO CENSORING
import surpyval

model = surpyval.Weibull.fit(x=duration_vec)
print(model)
print(model.aic())
model.plot()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


y = surpyval.Weibull.random(1000, 34, 0.63)
n = plt.hist(y, cumulative=True, bins=1000, log=True);


# In[ ]:




