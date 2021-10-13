import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.utils_filepaths import config_file_with_project_context
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, edit_config


def calc_dlc_to_tracklet_distances(dlc_tracks: pd.DataFrame,
                                   df_tracklet: pd.DataFrame,
                                   dlc_name: str,
                                   all_covering_ind: list,
                                   min_overlap: int = 5,
                                   min_dlc_confidence: float = 0.6):
    """For one DLC neuron, calculate distances between that track and all tracklets"""
    coords = ['z', 'x', 'y']

    this_dlc = dlc_tracks[dlc_name][coords]
    # Remove low confidence points
    #     low_conf = this_dlc['likelihood'] < min_dlc_confidence
    #     this_dlc.loc['x', low_conf].replace()

    all_tracklet_names = list(df_tracklet.columns.levels[0])
    all_dist = []
    for i, name in enumerate(tqdm(all_tracklet_names, leave=False)):
        # Check for already belonging to another track
        if i not in all_covering_ind:
            this_diff = df_tracklet[name][coords] - this_dlc

            # Check for enough common data points
            num_common_pts = this_diff['x'].notnull().sum()
            if num_common_pts >= min_overlap:
                dist = np.linalg.norm(this_diff, axis=1)
            else:
                dist = np.inf
        else:
            dist = np.inf

        all_dist.append(dist)

    return all_dist


def calc_covering_from_distances(all_dist, df_tracklet, used_indices, d_max=5, verbose=0):
    """Given distances between a dlc track and all tracklets, make a time-unique covering from the tracklets"""
    all_medians = list(map(np.nanmedian, all_dist))
    i_sorted_by_median_distance = np.argsort(all_medians)
    all_tracklet_names = list(df_tracklet.columns.levels[0])

    covering_ind = []
    covering_time_points = []
    t = df_tracklet.index
    these_dist = np.zeros_like(t, dtype=float)

    for i_tracklet in i_sorted_by_median_distance:
        # Check if this was used before
        if i_tracklet in used_indices:
            continue
        # Check distance; break becuase they are sorted by distance
        if all_medians[i_tracklet] > d_max:
            break
        # Check time overlap, except first time
        name = all_tracklet_names[i_tracklet]
        is_nan = df_tracklet[name]['x'].isnull()
        new_t = list(t[~is_nan])
        if len(covering_time_points) > 0:
            if any([t in covering_time_points for t in new_t]):
                continue

        # Save
        new_t = np.array(new_t)
        covering_time_points.extend(new_t)
        covering_ind.append(i_tracklet)
        these_dist[new_t] = all_dist[i_tracklet][new_t]

    if verbose >= 1:
        print(f"Covering of length {len(covering_time_points)} made from {len(covering_ind)} tracklets")
    if len(covering_time_points) == 0:
        logging.warning("No covering found, here are some diagnostics:")
        logging.warning(f"Minimum distance to other covering: {np.min(all_medians)}")
        logging.warning(f"Looped up to tracklet {i_tracklet} with distance {all_medians[i_tracklet]}")

    return covering_time_points, covering_ind, these_dist


def combine_matched_tracklets(these_tracklet_ind: list,
                              neuron_name: str,
                              df_tracklet: pd.DataFrame,
                              dlc_tracks: pd.DataFrame) -> pd.DataFrame:
    """Combines a covering of short tracklets and a gappy DLC track into a final DLC-style track"""
    coords = ['z', 'x', 'y', 'likelihood']

    # Extract the tracklets belonging to this neuron
    tracklet_names = df_tracklet.columns.levels[0]
    these_tracklet_names = [tracklet_names[i] for i in these_tracklet_ind]
    logging.info(f"Found {len(these_tracklet_names)} tracklets for neuron {neuron_name}")

    if len(these_tracklet_names) == 0:
        # Then no tracklets were found, so we pass an empty column
        one_col_shape = dlc_tracks[neuron_name].shape
        summed_tracklet_array = np.empty(one_col_shape)
        summed_tracklet_array[:] = np.nan

    else:
        # Combine tracklets (one dataframe, multiple names)
        new_df = df_tracklet[these_tracklet_names]
        all_arrs = []
        for c in coords:
            this_column = np.nansum([new_df[name][c] for name in these_tracklet_names], axis=0)
            # My tracker often does not track the very last frames, so fill with nan
            if len(this_column) < len(dlc_tracks):
                tmp = np.zeros(len(dlc_tracks) - len(this_column))
                tmp[:] = np.nan
                this_column = np.hstack([this_column, tmp])

            all_arrs.append(this_column)
        summed_tracklet_array = np.stack(all_arrs, axis=1)

    # Morph to DLC format
    cols = [[neuron_name], coords]
    cols = pd.MultiIndex.from_product(cols)
    summed_tracklet_df = pd.DataFrame(data=summed_tracklet_array, columns=cols)

    return summed_tracklet_df


def combine_all_dlc_and_tracklet_coverings(all_covering_ind: list,
                                           df_tracklet: pd.DataFrame,
                                           dlc_tracks: pd.DataFrame,
                                           rename_neurons: bool = False, verbose=0):
    """Combines coverings of all tracklets and DLC-tracked neurons"""
    all_df = []
    logging.info(f"Found {len(all_covering_ind)} tracklet-track combinations")

    # Build new tracklets into intermediate column
    all_neuron_names = dlc_tracks.columns.levels[0]
    for these_tracklet_ind, neuron_name in zip(all_covering_ind, all_neuron_names):
        df = combine_matched_tracklets(these_tracklet_ind, neuron_name, df_tracklet, dlc_tracks)
        all_df.append(df)
    new_tracklet_df = pd.concat(all_df, axis=1)
    if verbose >= 2:
        print(all_df)

    # Produce new dataframe
    all_neuron_names, final_track_df = combine_dlc_and_tracklets(all_neuron_names, dlc_tracks, new_tracklet_df)

    # To be sequential, and same as segmentation
    if rename_neurons:
        new_neuron_names = {name: f'neuron{i + 1}' for i, name in enumerate(all_neuron_names)}
        final_track_df.rename(columns=new_neuron_names, inplace=True)

    return final_track_df, new_tracklet_df


def combine_dlc_and_tracklets(all_neuron_names, dlc_tracks, new_tracklet_df):
    # Combine new and old tracklets (two dataframes, same neuron names)
    # Note: needs a loop because combine_first() doesn't work for multiindexes
    new_tracklet_df.replace(0, np.NaN, inplace=True)
    final_track_df = new_tracklet_df.copy()
    all_neuron_names = list(new_tracklet_df.columns.levels[0])
    for name in all_neuron_names:
        final_track_df[name] = new_tracklet_df[name].combine_first(dlc_tracks[name])
    return all_neuron_names, final_track_df


def combine_all_dlc_and_tracklet_coverings_from_config(track_config: config_file_with_project_context,
                                                       project_dir, DEBUG=False):
    """
    Improves tracking by combining DLC neurons with my short tracklets

    Parameters
    ----------
    track_config
    project_dir
    DEBUG

    Returns
    -------

    """

    d_max, df_dlc_tracks, df_tracklets, min_overlap, output_df_fname, rename_neurons = _unpack_tracklets_for_combining(
        project_dir, track_config)

    # Match tracklets to DLC neurons
    all_neuron_names = list(df_dlc_tracks.columns.levels[0])
    verbose = 0
    all_covering_ind = []
    used_indices = set()
    logging.info("Calculating distances between tracklets and DLC tracks")
    for i, dlc_name in enumerate(tqdm(all_neuron_names)):
        dist = calc_dlc_to_tracklet_distances(df_dlc_tracks, df_tracklets, dlc_name, all_covering_ind,
                                              min_overlap=min_overlap)
        out = calc_covering_from_distances(dist, df_tracklets, used_indices, d_max=d_max, verbose=verbose)
        # covering_time_points, covering_ind, these_dist = out
        _, covering_ind, _ = out
        all_covering_ind.append(covering_ind)
        used_indices.update(covering_ind)

        if DEBUG and i > 0:
            # Should do 2, so that the column concatenation is checked too
            logging.info("DEBUG: checking only 2 neurons")
            break

    # Combine and save
    # Rename to be sequential, like the reindexed segmentation
    logging.info("Concatenating tracklets")
    if DEBUG:
        rename_neurons = False
    combined_df, new_tracklet_df = combine_all_dlc_and_tracklet_coverings(all_covering_ind,
                                                                          df_tracklets,
                                                                          df_dlc_tracks,
                                                                          rename_neurons=rename_neurons,
                                                                          verbose=0)

    with safe_cd(project_dir):
        # Actually save
        combined_df.to_hdf(output_df_fname, key='df_with_missing')

        csv_fname = Path(output_df_fname).with_suffix('.csv')
        combined_df.to_csv(csv_fname)

        if not DEBUG:
            # Save only df_fname in yaml; don't overwrite other fields
            updates = {'final_3d_tracks_df': str(output_df_fname)}
            track_config.config.update(updates)
            track_config.update_on_disk()


def _unpack_tracklets_for_combining(project_dir, track_config):
    d_max = track_config.config['final_3d_postprocessing']['max_dist']
    min_overlap = track_config.config['final_3d_postprocessing']['min_overlap_dlc_and_tracklet']
    min_dlc_confidence = track_config.config['final_3d_postprocessing']['min_dlc_confidence']
    rename_neurons = track_config.config['final_3d_postprocessing']['rename_neurons']
    output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
    with safe_cd(project_dir):
        # TODO: add the tracklet fname to the config file
        tracklet_fname = os.path.join('2-training_data', 'all_tracklets.h5')
        dlc_fname = track_config.resolve_relative_path_from_config('final_3d_tracks_df')

        df_tracklets: pd.DataFrame = pd.read_hdf(tracklet_fname)
        df_dlc_tracks: pd.DataFrame = pd.read_hdf(dlc_fname)
        print(f"Combining {int(df_tracklets.shape[1]/4)} tracklets with {int(df_dlc_tracks.shape[1]/4)} neurons")
        df_dlc_tracks.replace(0, np.NaN, inplace=True)
    return d_max, df_dlc_tracks, df_tracklets, min_overlap, output_df_fname, rename_neurons
