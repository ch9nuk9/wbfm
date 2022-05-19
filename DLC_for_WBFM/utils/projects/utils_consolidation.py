import logging

import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.tracklets.high_performance_pandas import get_names_from_df
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


def consolidate_tracklets_using_manual_annotation(project_data: ProjectData):
    df_all_tracklets = project_data.df_all_tracklets
    unmatched_tracklet_names = get_names_from_df(df_all_tracklets)

    print(f"Original number of unique tracklets: {len(unmatched_tracklet_names)}")
    track_cfg = project_data.project_config.get_tracking_config()

    neurons_that_are_finished = project_data.get_list_of_finished_neurons()

    global2tracklet = project_data.global2tracklet
    # Build list of new consolidated tracklets
    consolidated_tracklets = []
    for neuron in tqdm(neurons_that_are_finished):
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

    output_df_fname = "df_consolidated.pickle"
    track_cfg.pickle_data_in_local_project(df_new, output_df_fname=output_df_fname, custom_writer=pd.to_pickle)
