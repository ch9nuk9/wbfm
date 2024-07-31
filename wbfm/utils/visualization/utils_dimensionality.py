import os
import plotly.express as px
import pandas as pd

from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation


def main():
    # Import traces (gcamp and immob)
    parent_folder = r'C:\Users\Charlie\Documents\traces_dataframes'
    fname = 'gcamp/data.h5'
    df_gcamp = pd.read_hdf(os.path.join(parent_folder, fname))
    fname = 'immob/data.h5'
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

    df['neuron_combined'] = df['neuron'].apply(_neuron_pair_name)
    df.dropna(inplace=True)
    # Only include neurons with more than the mininum neurons in both immob and gcamp
    min_neurons = 3
    neuron_counts_gcamp = df[df['dataset_type'] == 'gcamp']['neuron_combined'].value_counts()
    neuron_counts_immob = df[df['dataset_type'] == 'immob']['neuron_combined'].value_counts()
    enough_gcamp = neuron_counts_gcamp[neuron_counts_gcamp >= min_neurons].index
    enough_immob = neuron_counts_immob[neuron_counts_immob >= min_neurons].index
    enough_neurons = enough_gcamp.intersection(enough_immob)
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

    fig.show()


if __name__ == '__main__':
    main()
