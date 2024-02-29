import os

from wbfm.utils.general.hardcoded_paths import load_paper_datasets, get_hierarchical_modeling_dir
from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects, \
    build_behavior_time_series_from_multiple_projects


def export_data_for_hierarchical_model():
    """
    Loads the relevant projects, and exports both behavior and traces to a single .h5 file

    Returns
    -------

    """

    # Load projects
    all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])

    # Get individual data elements
    df_all_traces = build_trace_time_series_from_multiple_projects(all_projects_gcamp, use_paper_options=True)
    df_all_traces.sort_values(['dataset_name', 'local_time'], inplace=True)

    df_all_behavior = build_behavior_time_series_from_multiple_projects(all_projects_gcamp,
                                                                        behavior_names=['vb02_curvature', 'fwd',
                                                                                        'eigenworm0', 'eigenworm1',
                                                                                        'eigenworm2', 'eigenworm3'])
    df_all_behavior.sort_values(['dataset_name', 'local_time'], inplace=True)
    df_all_behavior['fwd'] = df_all_behavior['fwd'].astype(int)

    df_all_manifold = build_trace_time_series_from_multiple_projects(all_projects_gcamp,
                                                                     use_paper_options=True, residual_mode='pca_global')
    df_all_manifold.sort_values(['dataset_name', 'local_time'], inplace=True)

    # Align and export
    # Remake local time columns to just be integers
    df_all_traces['local_time'] = df_all_traces.groupby('dataset_name').cumcount()
    df_all_manifold['local_time'] = df_all_manifold.groupby('dataset_name').cumcount()
    df_all_behavior['local_time'] = df_all_behavior.groupby('dataset_name').cumcount()
    # Include all neurons
    df_all = df_all_traces.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                                 suffixes=('', '_manifold'))
    df_all = df_all.merge(df_all_behavior, on=['dataset_name', 'local_time'], how='inner')

    # Export
    fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
    df_all.to_hdf(fname, key='df_with_missing')
