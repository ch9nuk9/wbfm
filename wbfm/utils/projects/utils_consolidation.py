import logging
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, fill_missing_indices_with_nan
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df, get_next_name_generator
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once


def consolidate_tracklets_using_config(project_config: ModularProjectConfig,
                                       correct_only_finished_neurons=False,
                                       z_threshold=2,
                                       DEBUG=False):
    """
    Consolidates tracklets in all (or only finished) neurons into one large tracklet

    Resplit the tracklet if the change in z is above z_threshold

    Parameters
    ----------
    DEBUG
    project_config
    correct_only_finished_neurons
    z_threshold

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_config)

    df_all_tracklets = project_data.df_all_tracklets
    num_time_points = df_all_tracklets.shape[0]
    unmatched_tracklet_names = get_names_from_df(df_all_tracklets)

    print(f"Original number of unique tracklets: {len(unmatched_tracklet_names)}")
    track_cfg = project_data.project_config.get_tracking_config()

    if correct_only_finished_neurons:
        neuron_names = project_data.get_list_of_finished_neurons()
    else:
        neuron_names = get_names_from_df(project_data.final_tracks)

    # Generate names for new tracklets that don't conflict with the old ones
    name_gen = get_next_name_generator(df_all_tracklets, name_mode='tracklet')
    new_neuron2tracklets = dict()

    global2tracklet = project_data.global2tracklet
    # Build list of new consolidated tracklets
    consolidated_tracklets = []
    for neuron in tqdm(neuron_names):
        these_tracklets_names = global2tracklet[neuron]
        these_tracklets = [df_all_tracklets[n].dropna(axis=0) for n in these_tracklets_names]
        [unmatched_tracklet_names.remove(n) for n in these_tracklets_names]

        new_tracklet_name = next(name_gen)

        # Add new name in one line:
        # https://stackoverflow.com/questions/40225683/how-to-simply-add-a-column-level-to-a-pandas-dataframe
        joined_tracklet = pd.concat(these_tracklets, axis=0)
        joined_tracklet.columns = pd.MultiIndex.from_product([[new_tracklet_name], joined_tracklet.columns])

        # Check for duplicated indices... shouldn't happen, but humans can do it!
        idx_duplicated = joined_tracklet.index.duplicated(keep='first')
        if idx_duplicated.any():
            logging.warning(
                f"Found {sum(idx_duplicated)} duplicated indices in neuron {neuron}; keeping first instances")
            joined_tracklet = joined_tracklet[~idx_duplicated]

        # Make sure it is correctly indexed
        joined_tracklet.sort_index(inplace=True)
        joined_tracklet, num_added = fill_missing_indices_with_nan(joined_tracklet, expected_max_t=num_time_points)

        # Then resplit this single tracklet based on z_threshold and gaps (nan)
        df_diff = joined_tracklet[[(new_tracklet_name, 'z')]].diff().abs()
        split_list_dict = {new_tracklet_name: list(np.where(df_diff > z_threshold)[0])}
        block_starts, _ = get_contiguous_blocks_from_column(joined_tracklet[(new_tracklet_name, 'z')])
        if len(block_starts) > 0 and block_starts[0] == 0:
            block_starts = block_starts[1:]
        if len(block_starts) > 0:
            split_list_dict[new_tracklet_name].extend(block_starts)
            split_list_dict[new_tracklet_name].sort()
        if DEBUG:
            print(f"Splitting {new_tracklet_name} at {split_list_dict[new_tracklet_name]}, ({block_starts} from nan)")
            print(joined_tracklet)

        # Actually split
        df_split, _, name_mapping = split_all_tracklets_at_once(joined_tracklet, split_list_dict, name_gen=name_gen)
        if len(name_mapping) == 0:
            new_neuron2tracklets[neuron] = [new_tracklet_name]  # Unsplit
        else:
            new_neuron2tracklets[neuron] = name_mapping[new_tracklet_name]  # List of split names

        consolidated_tracklets.append(df_split)

        if DEBUG:
            break

    # Get remaining, unmatched tracklets
    df_unmatched = df_all_tracklets.loc[:, unmatched_tracklet_names]

    consolidated_tracklets.append(df_unmatched)
    df_new = pd.concat(consolidated_tracklets, axis=1)

    final_tracklet_names = get_names_from_df(df_new)
    print(f"Consolidated number of unique tracklets: {len(final_tracklet_names)}")

    # Save data
    if not DEBUG:
        output_df_fname = os.path.join("3-tracking", "postprocessing", "df_tracklets_consolidated.pickle")
        output_df_fname = track_cfg.pickle_data_in_local_project(df_new,
                                                                 relative_path=output_df_fname,
                                                                 make_sequential_filename=True,
                                                                 custom_writer=pd.to_pickle)
        output_neuron2tracklets_fname = os.path.join("3-tracking", "postprocessing", "global2tracklets_consolidated.pickle")
        output_neuron2tracklets_fname = track_cfg.pickle_data_in_local_project(new_neuron2tracklets,
                                                                               relative_path=output_neuron2tracklets_fname,
                                                                               make_sequential_filename=True)

        # Update config and filepaths
        output_neuron2tracklets_fname = track_cfg.unresolve_absolute_path(output_neuron2tracklets_fname)
        track_cfg.config['manual_correction_global2tracklet_fname'] = output_neuron2tracklets_fname
        output_df_fname = track_cfg.unresolve_absolute_path(output_df_fname)
        track_cfg.config['manual_correction_tracklets_df_fname'] = output_df_fname
        track_cfg.update_self_on_disk()
