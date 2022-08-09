import logging

import pandas as pd
from tqdm.auto import tqdm

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.projects.finished_project_data import ProjectData


def consolidate_tracklets_using_config(project_config: ModularProjectConfig,
                                       correct_only_finished_neurons=False):
    """
    Consolidates tracklets in all (or only finished) neurons into one large tracklet

    TODO: Optionally resplit the tracklets if there is a gap

    Parameters
    ----------
    project_config
    correct_only_finished_neurons

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_config)

    df_all_tracklets = project_data.df_all_tracklets
    unmatched_tracklet_names = get_names_from_df(df_all_tracklets)

    print(f"Original number of unique tracklets: {len(unmatched_tracklet_names)}")
    track_cfg = project_data.project_config.get_tracking_config()

    if correct_only_finished_neurons:
        neuron_names = project_data.get_list_of_finished_neurons()
    else:
        neuron_names = get_names_from_df(project_data.final_tracks)

    global2tracklet = project_data.global2tracklet
    # Build list of new consolidated tracklets
    consolidated_tracklets = []
    for neuron in tqdm(neuron_names):
        these_tracklets_names = global2tracklet[neuron]
        these_tracklets = [df_all_tracklets[n].dropna(axis=0) for n in these_tracklets_names]
        [unmatched_tracklet_names.remove(n) for n in these_tracklets_names]

        new_tracklet_name = these_tracklets_names[0]

        # Add new name in one line: https://stackoverflow.com/questions/40225683/how-to-simply-add-a-column-level-to-a-pandas-dataframe
        joined_tracklet = pd.concat(these_tracklets, axis=0)
        joined_tracklet.columns = pd.MultiIndex.from_product([[new_tracklet_name], joined_tracklet.columns])

        # Check for duplicated indices... shouldn't happen, but humans can do it!
        idx_duplicated = joined_tracklet.index.duplicated(keep='first')
        if idx_duplicated.any():
            logging.warning(
                f"Found {sum(idx_duplicated)} duplicated indices in neuron {neuron}; keeping first instances")
            joined_tracklet = joined_tracklet[~idx_duplicated]

        consolidated_tracklets.append(joined_tracklet)

    # Get remaining, unmatched tracklets
    df_unmatched = df_all_tracklets.loc[:, unmatched_tracklet_names]

    consolidated_tracklets.append(df_unmatched)
    df_new = pd.concat(consolidated_tracklets, axis=1)

    final_tracklet_names = get_names_from_df(df_new)
    print(f"Consolidated number of unique tracklets: {len(final_tracklet_names)}")

    output_df_fname = "df_tracklets_consolidated.pickle"
    output_df_fname = track_cfg.pickle_data_in_local_project(df_new, output_df_fname=output_df_fname, custom_writer=pd.to_pickle)
    track_cfg.config['manual_correction_tracklets_df_fname'] = output_df_fname
    track_cfg.update_self_on_disk()
