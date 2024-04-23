import os

import pandas as pd
from wbfm.utils.general.hardcoded_paths import load_paper_datasets, get_hierarchical_modeling_dir
from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects, \
    build_behavior_time_series_from_multiple_projects, build_cross_dataset_eigenworms, \
    build_pca_time_series_from_multiple_projects


def export_data_for_hierarchical_model(do_gfp=False, do_immobilized=False, skip_if_exists=True):
    """
    Loads the relevant projects, and exports both behavior and traces to a single .h5 file

    Returns
    -------

    """
    # Check if file exists
    data_dir = get_hierarchical_modeling_dir(do_gfp, do_immobilized)
    fname = os.path.join(data_dir, 'data.h5')
    if skip_if_exists and os.path.exists(fname):
        print(f"File {fname} already exists, skipping")
        return

    # Load projects
    if do_gfp:
        project_code = 'gfp'
    elif do_immobilized:
        project_code = 'immob'
    else:
        project_code = ['gcamp', 'hannah_O2_fm']
    all_projects = load_paper_datasets(project_code)

    # Get individual data elements
    df_all_traces = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_traces.sort_values(['dataset_name', 'local_time'], inplace=True)

    if not do_immobilized:
        df_all_behavior = build_behavior_time_series_from_multiple_projects(all_projects,
                                                                            behavior_names=['vb02_curvature', 'fwd'])
        df_all_behavior.sort_values(['dataset_name', 'local_time'], inplace=True)
        df_all_behavior['fwd'] = df_all_behavior['fwd'].astype(int)

        # Recalculate multi-dataset eigenworms
        df_eigenworms = build_cross_dataset_eigenworms(all_projects)

    # Get pca modes
    df_all_pca = build_pca_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_pca.rename(columns={i: f'pca_{i}' for i in range(4)}, inplace=True)

    df_all_manifold = build_trace_time_series_from_multiple_projects(all_projects,
                                                                     use_paper_options=True, residual_mode='pca_global')
    df_all_manifold.sort_values(['dataset_name', 'local_time'], inplace=True)

    # Align and export
    # Remake local time columns to just be integers
    df_all_traces['local_time'] = df_all_traces.groupby('dataset_name').cumcount()
    df_all_manifold['local_time'] = df_all_manifold.groupby('dataset_name').cumcount()
    if not do_immobilized:
        df_all_behavior['local_time'] = df_all_behavior.groupby('dataset_name').cumcount()
        df_eigenworms['local_time'] = df_eigenworms.groupby('dataset_name').cumcount()
    df_all_pca['local_time'] = df_all_pca.groupby('dataset_name').cumcount()
    # Include all neurons
    df_all = df_all_traces.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                                 suffixes=('', '_manifold'))
    if not do_immobilized:
        df_all = df_all.merge(df_all_behavior, on=['dataset_name', 'local_time'], how='inner')
        df_all = df_all.merge(df_eigenworms, on=['dataset_name', 'local_time'], how='inner')
    df_all = df_all.merge(df_all_pca, on=['dataset_name', 'local_time'], how='inner')

    # Export
    fname = os.path.join(data_dir, 'data.h5')
    df_all.to_hdf(fname, key='df_with_missing')


def export_data_for_oded_lab():
    """
    Uses the same data as export_data_for_hierarchical_model, but removes the manifold and pca columns

    Returns
    -------

    """
    # Load generated data
    data_dir = get_hierarchical_modeling_dir()
    fname = os.path.join(data_dir, 'data.h5')
    df_all = pd.read_hdf(fname)

    # Remove manifold and pca columns
    cols_to_remove = [col for col in df_all.columns if 'manifold' in col or 'pca' in col]
    cols_to_remove += ['vb02_curvature']
    df_all.drop(columns=cols_to_remove, inplace=True)

    # Export
    fname = os.path.join(data_dir, 'data_oded_lab.h5')
    df_all.to_hdf(fname, key='df_with_missing')

    return df_all


if __name__ == '__main__':
    export_data_for_hierarchical_model()
    export_data_for_hierarchical_model(do_gfp=True)
    export_data_for_hierarchical_model(do_immobilized=True)
