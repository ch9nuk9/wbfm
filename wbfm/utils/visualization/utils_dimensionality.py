import os
import plotly.express as px
import pandas as pd
from scipy import stats
import numpy as np
from tqdm.auto import tqdm

from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
from wbfm.utils.general.hardcoded_paths import (get_hierarchical_modeling_dir, role_of_neuron_dict,
                                                neurons_with_confident_ids)



def main(combine_left_right=True):
    # Import traces (gcamp and immob)
    parent_folder = get_hierarchical_modeling_dir()
    fname = 'data.h5'
    df_gcamp = pd.read_hdf(os.path.join(parent_folder, fname))
    parent_folder = get_hierarchical_modeling_dir(immobilized=True)
    fname = 'data.h5'
    df_immob = pd.read_hdf(os.path.join(parent_folder, fname))

    # Get the columns we want: variance explained by the manifold in %
    # First get the shared neuron names
    shared_neurons = df_gcamp.columns.intersection(df_immob.columns)
    nontarget_strings = ['neuron', 'manifold', 'pca', 'local_time', 'dataset']
    for string in nontarget_strings:
        shared_neurons = shared_neurons[~shared_neurons.str.contains(string)]

    # Get the target columns by calculating the variance explained by the raw trace and the manifold, and the ratio
    # But do this per dataset
    all_dfs = []
    df_gcamp_grouped = df_gcamp.groupby('dataset_name')
    df_immob_grouped = df_immob.groupby('dataset_name')

    for neuron in shared_neurons:
        # Gcamp
        var_raw = df_gcamp_grouped[neuron].var()
        var_manifold = df_gcamp_grouped[neuron + '_manifold'].var()
        ratio = var_manifold / var_raw
        # Make a small df from this series
        _df = pd.DataFrame(ratio, columns=['variance_explained'])
        _df['neuron'] = neuron
        _df['dataset_type'] = 'gcamp'
        all_dfs.append(_df)
        # Immob
        var_raw = df_immob_grouped[neuron].var()
        var_manifold = df_immob_grouped[neuron + '_manifold'].var()
        ratio = var_manifold / var_raw
        # Make a small df from this series
        _df = pd.DataFrame(ratio, columns=['variance_explained'])
        _df['neuron'] = neuron
        _df['dataset_type'] = 'immob'
        all_dfs.append(_df)

    # Create a dataframe and plot via plotly
    df = pd.concat(all_dfs)

    # New column for left/right combined neurons (remove 'L' or 'R' from the name, if it is the last character)
    def _neuron_pair_name(neuron_name):
        return neuron_name[:-1] if (neuron_name[-1] in ['L', 'R'] and len(neuron_name) > 3) else neuron_name

    if combine_left_right:
        df['neuron_combined'] = df['neuron'].apply(_neuron_pair_name)
    else:
        df['neuron_combined'] = df['neuron']
    df.dropna(inplace=True)
    # Only include neurons with more than the mininum neurons in both immob and gcamp
    min_neurons = 3
    neuron_counts_gcamp = df[df['dataset_type'] == 'gcamp']['neuron_combined'].value_counts()
    neuron_counts_immob = df[df['dataset_type'] == 'immob']['neuron_combined'].value_counts()
    enough_gcamp = neuron_counts_gcamp[neuron_counts_gcamp >= min_neurons].index
    enough_immob = neuron_counts_immob[neuron_counts_immob >= min_neurons].index
    enough_neurons = enough_gcamp.intersection(enough_immob).intersection(neurons_with_confident_ids())
    df = df[df['neuron_combined'].isin(enough_neurons)]

    # Sort by mean variance explained in immob (not gcamp) per neuron
    # df_grouped = df[df['dataset_type'] == 'immob'].groupby('neuron_combined')
    # mean_var_explained = df_grouped['variance_explained'].mean()
    # # copy mean_var_explained to all rows of the same neuron, including gcamp
    # df = df.set_index('neuron_combined')
    # df['mean_var_explained'] = mean_var_explained
    # df = df.sort_values(by='mean_var_explained', ascending=True).reset_index()

    # Sort by difference in variance explained between immob and gcamp
    df_grouped = df[df['dataset_type'] == 'gcamp'].groupby('neuron_combined')
    mean_var_explained_gcamp = df_grouped['variance_explained'].median()
    df_grouped = df[df['dataset_type'] == 'immob'].groupby('neuron_combined')
    mean_var_explained_immob = df_grouped['variance_explained'].median()
    diff_var_explained = mean_var_explained_immob - mean_var_explained_gcamp
    df = df.set_index('neuron_combined')
    df['diff_var_explained'] = diff_var_explained
    df = df.sort_values(by='diff_var_explained', ascending=True).reset_index()

    fig = px.box(df, x='neuron_combined', y='variance_explained', color='dataset_type', #points='all',
                 color_discrete_map=plotly_paper_color_discrete_map(),
                 title='Variance explained by the manifold')
    # Add t-test annotation
    # add_p_value_annotation(fig, x_label='all', show_only_stars=True, show_ns=False)
    # apply_figure_settings(fig)
    #

    # fig.show(renderer='browser')

    # Calculate the p values and effect sizes of the difference in variance explained between immob and gcamp
    # for each neuron
    p_values = {}
    effect_sizes = {}
    permutations = 10000
    for neuron in tqdm(enough_neurons):
        df_neuron = df[df['neuron_combined'] == neuron]
        immob = df_neuron[df_neuron['dataset_type'] == 'immob']['variance_explained']
        gcamp = df_neuron[df_neuron['dataset_type'] == 'gcamp']['variance_explained']
        p_value = stats.ttest_ind(immob, gcamp, equal_var=False, random_state=4242)[1]# permutations=permutations)[1]
        effect_size = immob.median() - gcamp.median()
        p_values[neuron] = p_value
        effect_sizes[neuron] = effect_size

    # Second plot: volcano plot of p values and effect sizes
    df_p_values = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value'])
    df_effect_sizes = pd.DataFrame.from_dict(effect_sizes, orient='index', columns=['effect_size'])
    df_combined = df_p_values.join(df_effect_sizes)

    bonferroni_factor = len(enough_neurons)
    df_combined['significant'] = (df_combined['p_value'] * bonferroni_factor) < 0.05
    df_combined['minus_log_p'] = -np.log10(df_combined['p_value'] + 1e-6)
    df_combined['neuron_name'] = df_combined.index
    df_combined['role'] = df_combined['neuron_name'].map(role_of_neuron_dict())
    df_combined['fwd_rev'] = df_combined['neuron_name'].map(lambda x: role_of_neuron_dict(only_fwd_rev=True).get(x, ''))
    # Concatenate columns
    df_combined['combined_role'] = df_combined['role'] + ', ' + df_combined['fwd_rev']
    mapping = {'Interneuron, Forward': 'red',
               'Interneuron, Reverse': 'blue',
               'Motor, Forward': 'lightpurple',
               'Motor, Reverse': 'lightpurple',
               'Sensory, ': 'gray', 'Interneuron, ': 'lightgray', 'Motor, ': 'lightgray'}
    print(df_combined)
    fig = px.scatter(df_combined, x='effect_size', y='minus_log_p',
                     # symbol='role', color='fwd_rev',
                     color='combined_role',
                     color_discrete_map=mapping,
                     hover_name=df_combined.index,
                     title='Effect sizes and p values of the difference in variance explained between immob and gcamp')

    apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)
    # apply_figure_settings(fig, width_factor=0.3, height_factor=0.3)

    fig.update_traces(marker=dict(size=12))
    fig.show(renderer='browser')


if __name__ == '__main__':
    main(combine_left_right=False)
