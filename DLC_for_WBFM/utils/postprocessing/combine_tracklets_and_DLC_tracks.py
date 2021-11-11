import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numba
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import fix_matches_to_use_keys_not_int, \
    fix_global2tracklet_full_dict
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from scipy.spatial.distance import squareform, pdist
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.utils_filepaths import SubfolderConfigFile, read_if_exists, ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd, get_sequential_filename


def calc_dlc_to_tracklet_distances(this_global_track: np.ndarray,
                                   list_tracklets_zxy: list,
                                   used_indices: set,
                                   min_overlap: int = 5):
    """For one DLC neuron, calculate distances between that track and all tracklets"""

    # all_dist = np.zeros(len(list_tracklets_zxy))
    all_dist = []
    for i, this_tracklet in enumerate(list_tracklets_zxy):
        # Check for already belonging to another track
        if i not in used_indices:
            dist = calc_dist_if_overlap(this_tracklet, min_overlap, this_global_track)
        else:
            dist = np.inf
        all_dist.append(dist)
    return all_dist


def calc_dist_if_overlap(this_tracklet: np.ndarray, min_overlap: int, this_global_track: np.ndarray):
    this_diff = this_tracklet - this_global_track

    # Check for enough common data points
    # num_common_pts = this_diff['x'].notnull().sum()
    num_common_pts = np.count_nonzero(~np.isnan(this_diff[:, 0]))
    if num_common_pts >= min_overlap:
        dist = np.linalg.norm(this_diff, axis=1)
    else:
        dist = np.inf
    return dist


def OLD_calc_dlc_to_tracklet_distances(dlc_tracks: pd.DataFrame,
                                   df_tracklet: pd.DataFrame,
                                   dlc_name: str,
                                   used_indices: set,
                                   min_overlap: int = 5,
                                   min_dlc_confidence: float = 0.6):
    """For one DLC neuron, calculate distances between that track and all tracklets"""
    coords = ['z', 'x', 'y']

    # TODO: use confidence of dlc tracks
    this_dlc = dlc_tracks[dlc_name][coords]
    # Remove low confidence points
    #     low_conf = this_dlc['likelihood'] < min_dlc_confidence
    #     this_dlc.loc['x', low_conf].replace()

    all_tracklet_names = list(df_tracklet.columns.levels[0])
    all_dist = []
    for i, name in enumerate(tqdm(all_tracklet_names, leave=False)):
        # Check for already belonging to another track
        if i not in used_indices:
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


def calc_covering_from_distances(all_dist, df_tracklet, used_indices,
                                 covering_time_points=None,
                                 d_max=5, verbose=0):
    """Given distances between a dlc track and all tracklets, make a time-unique covering from the tracklets"""
    if covering_time_points is None:
        covering_time_points = []
    all_medians = list(map(np.nanmedian, all_dist))
    i_sorted_by_median_distance = np.argsort(all_medians)
    all_tracklet_names = list(df_tracklet.columns.levels[0])

    covering_tracklet_ind = []
    t = df_tracklet.index
    assert len(t) == int(t[-1])+1, "Tracklet dataframe has missing indices, and will cause errors"
    these_dist = np.zeros_like(t, dtype=float)

    for i_tracklet in i_sorted_by_median_distance:
        # Check if this was used before
        if i_tracklet in used_indices:
            continue
        # Check distance; break because they are sorted by distance
        if all_medians[i_tracklet] > d_max:
            break
        # Check time overlap, except first time
        name = all_tracklet_names[i_tracklet]
        is_nan = df_tracklet[name]['x'].isnull()
        newly_covered_times = list(t[~is_nan])
        if len(covering_time_points) > 0:
            if any([t in covering_time_points for t in newly_covered_times]):
                continue

        # Save if the tracklet passes all conditions above
        newly_covered_times = np.array(newly_covered_times)
        covering_time_points.extend(newly_covered_times)
        covering_tracklet_ind.append(i_tracklet)
        these_dist[newly_covered_times] = all_dist[i_tracklet][newly_covered_times]

    if verbose >= 1:
        print(f"Covering of length {len(covering_time_points)} made from {len(covering_tracklet_ind)} tracklets")
    if len(covering_time_points) == 0:
        logging.warning("No covering found, here are some diagnostics:")
        # logging.warning(f"Minimum distance to other covering: {np.min(all_medians)}")
        logging.warning(f"Looped up to tracklet {i_tracklet} with distance {all_medians[i_tracklet]}")

    return covering_time_points, covering_tracklet_ind, these_dist


def combine_matched_tracklets(these_tracklet_names: List[str],
                              neuron_name: str,
                              df_tracklet: pd.DataFrame,
                              dlc_tracks: pd.DataFrame,
                              verbose=1) -> pd.DataFrame:
    """Combines a covering of short tracklets and a gappy DLC track into a final DLC-style track"""
    coords = ['z', 'x', 'y', 'likelihood']

    # these_tracklet_names = fix_matches_to_use_keys_not_int(df_tracklet, these_tracklet_ind)
    logging.info(f"Found {len(these_tracklet_names)} tracklets for {neuron_name}")

    if len(these_tracklet_names) == 0:
        # Then no tracklets were found, so we pass an empty column
        one_col_shape = dlc_tracks[neuron_name].shape
        summed_tracklet_array = np.empty(one_col_shape)
        summed_tracklet_array[:] = np.nan
        if verbose >= 1:
            logging.warning(f"No tracklets found for {neuron_name}")

    else:
        # Combine tracklets (one dataframe, multiple names)
        new_df = df_tracklet[these_tracklet_names].copy()
        numpy_columns = []
        for c in coords:
            this_column = np.nansum([new_df[name][c] for name in these_tracklet_names], axis=0)
            # My tracker often does not track the very last frames, so fill with nan
            if len(this_column) < len(dlc_tracks):
                tmp = np.zeros(len(dlc_tracks) - len(this_column))
                tmp[:] = np.nan
                this_column = np.hstack([this_column, tmp])

            numpy_columns.append(this_column)
            if verbose >= 1 and c == 'z':
                logging.info(f"Matching neuron {neuron_name} with tracklets {these_tracklet_names}")
                logging.info(f"summed_tracklet_array: {this_column}")
                logging.info(f"Nonzero entries: {np.where(this_column>0)}")
        summed_tracklet_array = np.stack(numpy_columns, axis=1)


    # Morph to DLC format
    cols = [[neuron_name], coords]
    cols = pd.MultiIndex.from_product(cols)
    summed_tracklet_df = pd.DataFrame(data=summed_tracklet_array, columns=cols)

    return summed_tracklet_df


def combine_global_and_tracklet_coverings(global2tracklet: Dict[str, List[str]],
                                          df_tracklet: pd.DataFrame,
                                          df_global_tracks: pd.DataFrame,
                                          keep_only_tracklets_in_final_tracks: bool,
                                          verbose=0):
    """
    Combines coverings of all tracklets and DLC-tracked neurons

    If there is a tracklet covering for a time point, then it overwrites the global track.
    If keep_only_tracklets_in_final_tracks is false, then points without a tracklet are filled by the global
        track position, if any. Otherwise, only tracklets survive to the final dataframe
    """
    all_df = []
    logging.info(f"Found {len(global2tracklet)} tracklet-track combinations")

    # Build new tracklets into intermediate column
    for neuron_name, these_tracklet_names in global2tracklet.items():
        df = combine_matched_tracklets(these_tracklet_names, neuron_name, df_tracklet, df_global_tracks,
                                       verbose=verbose-1)
        all_df.append(df)
    new_tracklet_df = pd.concat(all_df, axis=1)
    if verbose >= 3:
        print(all_df)

    # Produce new dataframe
    if not keep_only_tracklets_in_final_tracks:
        final_track_df = combine_dlc_and_tracklets(new_tracklet_df, df_global_tracks)
    else:
        final_track_df = new_tracklet_df

    return final_track_df, new_tracklet_df


def combine_dlc_and_tracklets(new_tracklet_df, dlc_tracks):
    # Combine new and old tracklets (two dataframes, same neuron names)
    # Note: needs a loop because combine_first() doesn't work for multiindexes
    new_tracklet_df.replace(0, np.NaN, inplace=True)
    final_track_df = new_tracklet_df.copy()
    all_neuron_names = list(new_tracklet_df.columns.levels[0])
    for name in all_neuron_names:
        final_track_df[name] = new_tracklet_df[name].combine_first(dlc_tracks[name])
    return final_track_df


def combine_all_dlc_and_tracklet_coverings_from_config(track_config: SubfolderConfigFile,
                                                       training_cfg: SubfolderConfigFile,
                                                       project_cfg: ModularProjectConfig,
                                                       use_imputed_df=False,
                                                       use_manual_matches=True,
                                                       DEBUG=False):
    """
    Improves tracking by combining DLC neurons with my short tracklets

    Parameters
    ----------
    track_config
    training_cfg
    project_dir
    DEBUG

    Returns
    -------

    """

    d_max, df_global_tracks, df_tracklets, min_overlap, output_df_fname, \
        keep_only_tracklets_in_final_tracks, global2tracklet, used_indices = _unpack_tracklets_for_combining(
            project_cfg, training_cfg, track_config, use_imputed_df, use_manual_matches)

    # Match tracklets to DLC neurons
    global_neuron_names = list(df_global_tracks.columns.levels[0])

    logging.info(f"{len(used_indices)} / {df_tracklets.shape[1]} tracklets matched from previous analysis")
    verbose = 0

    # Pre-make coordinates so that the dataframe is not continuously indexed
    coords = ['z', 'x', 'y']
    all_tracklet_names = list(df_tracklets.columns.levels[0])
    list_tracklets_zxy = [df_tracklets[name][coords].to_numpy() for name in all_tracklet_names]

    logging.info("Calculating distances between tracklets and DLC tracks")
    for i, global_name in enumerate(tqdm(global_neuron_names)):

        # TODO: use confidence of dlc tracks
        this_global_track = df_global_tracks[global_name][coords].to_numpy()
        # TODO: make the tracklets the proper length before this
        this_global_track = this_global_track[:-1, :]
        dist = calc_dlc_to_tracklet_distances(this_global_track, list_tracklets_zxy, used_indices,
                                              min_overlap=min_overlap)
        previous_matches = global2tracklet[global_name]
        covering_time_points = get_already_covered_indices(df_tracklets, previous_matches)
        out = calc_covering_from_distances(dist, df_tracklets, used_indices,
                                           covering_time_points=covering_time_points,
                                           d_max=d_max, verbose=verbose)
        # covering_time_points, covering_ind, these_dist = out
        _, covering_ind, _ = out
        # all_covering_ind.append(covering_ind)
        global2tracklet[global_name].extend(covering_ind)
        used_indices.update(covering_ind)

        if DEBUG and i > 0:
            # Should do 2, so that the column concatenation is checked too
            logging.info("DEBUG: checking only 2 neurons")
            break

    logging.info(f"{len(used_indices)} / {df_tracklets.shape[1]} tracklets used in total")

    _save_tracklet_matches(global2tracklet, project_cfg.project_dir, track_config)

    # Combine and save
    # final_tracks_from_tracklet_matches(DEBUG, df_global_tracks, df_tracklets, global2tracklet,
    #                                    keep_only_tracklets_in_final_tracks, output_df_fname, project_dir, track_config)


def final_tracks_from_tracklet_matches_from_config(track_config: SubfolderConfigFile,
                                                   training_cfg: SubfolderConfigFile,
                                                   project_cfg: ModularProjectConfig,
                                                   use_imputed_df,
                                                   use_manual_matches,
                                                   DEBUG=False):

    d_max, df_global_tracks, df_tracklets, min_overlap, output_df_fname, \
        keep_only_tracklets_in_final_tracks, global2tracklet, used_indices = _unpack_tracklets_for_combining(
            project_cfg, training_cfg, track_config, use_imputed_df, use_manual_matches)

    # Rename to be sequential, like the reindexed segmentation
    logging.info("Concatenating tracklets")
    combined_df, new_tracklet_df = combine_global_and_tracklet_coverings(global2tracklet,
                                                                         df_tracklets,
                                                                         df_global_tracks,
                                                                         keep_only_tracklets_in_final_tracks,
                                                                         verbose=0)
    _save_combined_dataframe(DEBUG, combined_df, output_df_fname, project_cfg.project_dir, track_config)


def get_already_covered_indices(df_tracklets, previous_matches):
    if len(previous_matches) > 0:
        all_tracklet_names = list(df_tracklets.columns.levels[0])
        all_tracklet_ind = df_tracklets.index
        covering_time_points = []
        for i2 in previous_matches:
            if type(i2) == str:
                tracklet_name = i2
            else:
                tracklet_name = all_tracklet_names[i2]
            is_nan = df_tracklets[tracklet_name]['x'].isnull()
            covering_time_points.extend(list(all_tracklet_ind[~is_nan]))
    else:
        covering_time_points = None
    return covering_time_points


def _save_combined_dataframe(DEBUG, combined_df, output_df_fname, project_dir, track_config):
    with safe_cd(project_dir):
        # Actually save
        logging.info(f"Saving to: {output_df_fname}")
        combined_df.to_hdf(output_df_fname, key='df_with_missing')

        csv_fname = Path(output_df_fname).with_suffix('.csv')
        combined_df.to_csv(csv_fname)

        if not DEBUG:
            # Save only df_fname in yaml; don't overwrite other fields
            updates = {'final_3d_tracks_df': str(output_df_fname)}
            track_config.config.update(updates)
            track_config.update_on_disk()


def _save_tracklet_matches(global2tracklet, project_dir, track_config):
    with safe_cd(project_dir):
        abs_fname = track_config.resolve_relative_path_from_config('global2tracklet_matches_fname')
        abs_fname = get_sequential_filename(abs_fname)
        track_config.pickle_in_local_project(global2tracklet, abs_fname)

        rel_fname = track_config.unresolve_absolute_path(abs_fname)
        track_config.config.update({'global2tracklet_matches_fname': rel_fname})
        track_config.update_on_disk()


def _unpack_tracklets_for_combining(project_cfg: ModularProjectConfig,
                                    training_cfg: SubfolderConfigFile,
                                    track_config: SubfolderConfigFile,
                                    use_imputed_df,
                                    use_manual_matches):
    d_max = track_config.config['final_3d_postprocessing']['max_dist']
    min_overlap = track_config.config['final_3d_postprocessing']['min_overlap_dlc_and_tracklet']
    min_dlc_confidence = track_config.config['final_3d_postprocessing']['min_dlc_confidence']
    keep_only_tracklets_in_final_tracks = track_config.config['final_3d_postprocessing'][
        'keep_only_tracklets_in_final_tracks']
    output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
    output_df_fname = get_sequential_filename(output_df_fname)

    # Use main object to load
    project_data = ProjectData(project_cfg.project_dir, project_cfg)
    df_tracklets = project_data.df_all_tracklets
    if not use_manual_matches:
        project_data.precedence_global2tracklet = ['automatic', 'manual']

    with safe_cd(project_cfg.project_dir):

        subfolder = os.path.join('3-tracking', 'postprocessing')
        Path(subfolder).mkdir(exist_ok=True)
        # TODO: add the tracklet fname to the config file
        # tracklet_fname = training_cfg.resolve_relative_path('all_tracklets.h5', prepend_subfolder=True)
        if not use_imputed_df:
            global_tracks_fname = track_config.resolve_relative_path_from_config('final_3d_tracks_df')
        else:
            global_tracks_fname = track_config.resolve_relative_path_from_config('missing_data_imputed_df')
            logging.info(f"Reading from the imputed data, not the original global tracks: {global_tracks_fname}")

        # df_tracklets: pd.DataFrame = read_if_exists(tracklet_fname)
        df_global_tracks: pd.DataFrame = read_if_exists(global_tracks_fname)
        logging.info(f"Combining {int(df_tracklets.shape[1]/4)} tracklets with {int(df_global_tracks.shape[1]/4)} neurons")
        df_global_tracks.replace(0, np.NaN, inplace=True)

    # Check for previous matches, and start from them
    global2tracklet = project_data.global2tracklet
    if global2tracklet is None:
        logging.info(f"Did not find previous tracklet matches")
        global2tracklet = defaultdict(list)
        used_indices = set()
    else:
        logging.info(f"Found previous tracklet matches")
        used_indices = set()
        [used_indices.update(ind) for ind in global2tracklet.values()]

    # TODO: don't allow these to be integers from the beginning
    global2tracklet = fix_global2tracklet_full_dict(df_tracklets, global2tracklet)

    # fname = track_config.resolve_relative_path_from_config('global2tracklet_matches_fname')
    # if Path(fname).exists():
    #     logging.info(f"Found previous tracklet matches at {fname}")
    #     global2tracklet = pickle_load_binary(fname)
    #     used_indices = set()
    #     [used_indices.update(ind) for ind in global2tracklet.values()]
    # else:
    #     logging.info(f"Did not find previous tracklet matches")
    #     global2tracklet = defaultdict(list)
    #     used_indices = set()

    return d_max, df_global_tracks, df_tracklets, min_overlap, output_df_fname, \
        keep_only_tracklets_in_final_tracks, global2tracklet, used_indices


def remove_overmatching(df, tol=1e-3):
    """
    Checks the tracklet + tracks combined dataframe for multiple neuron names being assigned to the same location, and
    removes the one with lower confidence

    Returns
    -------

    """
    all_neurons = list(df.columns.levels[0])
    coords = ['z', 'x', 'y']
    tspan = list(df.index)

    for i_time in tqdm(tspan):
        all_zxy = np.array([df[n][coords].iloc[i_time].to_numpy() for n in all_neurons])
        dist_mat = squareform(pdist(all_zxy))
        dist_mat[np.diag_indices_from(dist_mat)] = np.nan

        min_dist = np.nanmin(dist_mat, axis=1)
        all_overmatched = []
        # Get neurons that are extremely close
        for i0, d in enumerate(min_dist):
            if np.abs(d) < tol:
                i1 = np.nanargmin(dist_mat[i0, :])
                all_overmatched.append([all_neurons[i0], all_neurons[i1]])

        # Remove the neuron with lower confidence
        for n0, n1 in all_overmatched:
            conf0, conf1 = df[n0]['likelihood'].iloc[i_time], df[n1]['likelihood'].iloc[i_time]
            d_conf = conf0 - conf1
            if np.isnan(d_conf):
                continue
            if d_conf == 0:
                raise ValueError
            if d_conf > 0:
                df[n1].iloc[i_time] = np.nan
            else:
                df[n0].iloc[i_time] = np.nan

    return df


def remove_overmatched_tracks_using_config(track_cfg: SubfolderConfigFile):

    df_fname = track_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
    df = read_if_exists(df_fname)

    df = remove_overmatching(df)

    # Overwrite
    df.to_hdf(df_fname, key='df_with_missing')
    df.to_csv(Path(df_fname).with_suffix('.csv.'))
