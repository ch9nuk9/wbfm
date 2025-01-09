import os

import pandas as pd

from wbfm.utils.general.hardcoded_paths import load_paper_datasets, get_hierarchical_modeling_dir
from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects, \
    build_behavior_time_series_from_multiple_projects, build_cross_dataset_eigenworms, \
    build_pca_time_series_from_multiple_projects


def export_data_for_hierarchical_model(suffix='', skip_if_exists=True, delete_if_exists=False):
    """
    Loads the relevant projects, and exports both behavior and traces to a single .h5 file

    Returns
    -------

    """
    # Check if file exists
    data_dir = get_hierarchical_modeling_dir(suffix=suffix)
    output_fname = os.path.join(data_dir, 'data.h5')
    print(f"Exporting data to {output_fname} with suffix {suffix}")
    if os.path.exists(output_fname):
        if skip_if_exists:
            print(f"File {output_fname} already exists, skipping")
            return
        elif delete_if_exists:
            print(f"File {output_fname} already exists, deleting")
            os.remove(output_fname)
        else:
            raise FileExistsError(f"File {output_fname} already exists; set delete_if_exists=True to overwrite"
                                  f" or skip_if_exists=True to skip")

    # Load projects from the suffix
    all_projects = load_paper_datasets(suffix)
    do_immobilized = 'immob' in suffix

    # Get individual data elements
    df_all_traces = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_traces.sort_values(['dataset_name', 'local_time'], inplace=True)

    if not do_immobilized:
        behavior_names = ['curvature_vb02', 'curvature_5', 'curvature_10', 'curvature_15', 'curvature_20',
                          'fwd', 'speed', 'ventral_only_head_curvature', 'dorsal_only_head_curvature',
                          'ventral_only_body_curvature', 'dorsal_only_body_curvature', 'self_collision',
                          'head_signed_curvature', 'summed_curvature',
                          'worm_nose_peak_frequency', 'worm_head_peak_frequency', 'worm_body_peak_frequency']
        df_all_behavior = build_behavior_time_series_from_multiple_projects(all_projects, behavior_names=behavior_names)
        df_all_behavior.sort_values(['dataset_name', 'local_time'], inplace=True)
        df_all_behavior['fwd'] = df_all_behavior['fwd'].astype(int)

        # Recalculate multi-dataset eigenworms
        df_eigenworms = build_cross_dataset_eigenworms(all_projects)

    # Get pca modes
    df_all_pca = build_pca_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_pca.rename(columns={i: f'pca_{i}' for i in range(4)}, inplace=True)

    # Get manifold in two ways: pc1 subtraction and pc1 and 2 subtraction (original)
    df_all_manifold = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True,
                                                                     interpolate_nan=True, residual_mode='pca_global')
    df_all_manifold.sort_values(['dataset_name', 'local_time'], inplace=True)
    # New
    df_all_manifold1 = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True,
                                                                      interpolate_nan=True,
                                                                      residual_mode='pca_global_1')
    df_all_manifold1.sort_values(['dataset_name', 'local_time'], inplace=True)

    # Align and export
    # Remake local time columns to just be integers
    df_all_traces['local_time'] = df_all_traces.groupby('dataset_name').cumcount()
    df_all_manifold['local_time'] = df_all_manifold.groupby('dataset_name').cumcount()
    df_all_manifold1['local_time'] = df_all_manifold1.groupby('dataset_name').cumcount()
    if not do_immobilized:
        df_all_behavior['local_time'] = df_all_behavior.groupby('dataset_name').cumcount()
        df_eigenworms['local_time'] = df_eigenworms.groupby('dataset_name').cumcount()
    df_all_pca['local_time'] = df_all_pca.groupby('dataset_name').cumcount()
    # Include all neurons
    df_all = df_all_traces.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                                 suffixes=('', '_manifold'))
    df_all = df_all.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                          suffixes=('', '_manifold1'))
    if not do_immobilized:
        df_all = df_all.merge(df_all_behavior, on=['dataset_name', 'local_time'], how='inner')
        df_all = df_all.merge(df_eigenworms, on=['dataset_name', 'local_time'], how='inner')
    df_all = df_all.merge(df_all_pca, on=['dataset_name', 'local_time'], how='inner')

    # Export
    df_all.to_hdf(output_fname, key='df_with_missing')
    print(f"Exported to {output_fname}")


def get_dataframe_for_single_neuron(Xy, neuron_name, curvature_terms=None, dataset_name='all', additional_columns=None,
                                    residual_mode='pca_global', verbose=1):
    if verbose >= 1:
        print(f"Found data columns: {Xy.columns} and datasets: {Xy['dataset_name'].unique()}")
        print(f"Attempting to load curvature terms {curvature_terms} and additional columns {additional_columns}")

    if dataset_name != 'all':
        _Xy = Xy[Xy['dataset_name'] == dataset_name]
    else:
        _Xy = Xy
    if curvature_terms is None:
        curvature_terms = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    # First, extract data, z-score, and drop na values
    # Allow gating based on the global component of the neuron itself (not used)
    x = _Xy[f'{neuron_name}_manifold']
    x = (x - x.mean()) / x.std()  # z-score
    # Alternative: include the pca modes (currently used)
    x_pca0 = _Xy[f'pca_0']
    x_pca0 = (x_pca0 - x_pca0.mean()) / x_pca0.std()  # z-score
    x_pca1 = _Xy[f'pca_1']
    x_pca1 = (x_pca1 - x_pca1.mean()) / x_pca1.std()  # z-score
    if residual_mode == 'pca_global' or residual_mode == 'pca_global_2':
        # Predict the residual
        y = _Xy[f'{neuron_name}'] - _Xy[f'{neuron_name}_manifold']
    elif residual_mode == 'pca_global_1':
        # Subtract only pc1
        y = _Xy[f'{neuron_name}'] - _Xy[f'{neuron_name}_manifold1']
    elif residual_mode is None:
        y = _Xy[f'{neuron_name}']
    else:
        raise ValueError(f"Unknown residual mode {residual_mode}; should be None, 'pca_global', or 'pca_global_1'")
    y = (y - y.mean()) / y.std()  # z-score
    if y.std() == 0:
        raise ValueError(f"Standard deviation of y is 0 for {neuron_name} in {dataset_name} and residual_mode {residual_mode}... "
                         f"This could be due to no data, or a bug in the residual calculation")
    # Interesting covariate
    curvature = _Xy[curvature_terms]
    curvature = (curvature - curvature.mean()) / curvature.std()  # z-score
    # State
    fwd = _Xy['fwd'].astype(str)
    # Package as dataframe again, and drop na values
    all_dfs = [pd.DataFrame({'y': y, 'x': x, 'x_pca0': x_pca0, 'x_pca1': x_pca1,
                             'dataset_name': _Xy['dataset_name'], 'fwd': fwd}),
               pd.DataFrame(curvature)]
    if additional_columns is not None:
        all_dfs.append(_Xy[additional_columns])
    df_model = pd.concat(all_dfs, axis=1)
    if verbose >= 1:
        print(f"Number of non-nan values per column: {df_model.count()}")
    df_model = df_model.dropna()
    if verbose >= 1:
        print(f"Loaded {df_model.shape[0]} samples for {neuron_name} in {dataset_name}")
    return df_model
