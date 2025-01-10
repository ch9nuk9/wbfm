import logging
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture, WormReferencePosture
from wbfm.utils.general.high_performance_pandas import get_names_from_df, empty_dataframe_like
from wbfm.utils.neuron_matching.class_frame_pair import calc_FramePair_from_Frames
from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence
from wbfm.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons, TrackedWorm
from wbfm.utils.general.distance_functions import summarize_confidences_outlier_percent, precalculate_lists_from_dataframe, \
    calc_global_track_to_tracklet_distances_subarray
from wbfm.utils.projects.finished_project_data import ProjectData
import networkx as nx

from tqdm.auto import tqdm


def long_range_matches_from_config(project_path, to_save=True, verbose=2):
    # project_path = "/home/charles/dlc_stacks/worm3-newseg-2021_11_17/project_config.yaml"

    project_data = ProjectData.load_final_project_data_from_config(project_path, to_load_tracklets=True, to_load_frames=True)
    df_tracklets = project_data.df_all_tracklets
    segmentation_metadata = project_data.segmentation_metadata
    all_frames = project_data.raw_frames
    all_matches = project_data.raw_matches
    # raw_clust = project_data.raw_clust

    frame_pair_options = all_matches[(0, 1)].options

    fname = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv"
    fname_X = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_X_coords.csv"
    fname_Y = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_Y_coords.csv"

    full_posture = WormFullVideoPosture(fname, fname_X, fname_Y)
    reference_posture = WormReferencePosture(0, full_posture)

    # Initialize TrackedNeurons at 0, and initialize the TrackedWorm
    # Get all tracklets that start at t=0
    all_tracklet_names = df_tracklets.columns.get_level_values(0).drop_duplicates()

    worm_obj = initialize_worm_object(df_tracklets, segmentation_metadata)

    all_long_range_matches = extend_tracks_using_similar_postures(all_frames, frame_pair_options,
                                                                  reference_posture, verbose, worm_obj)

    global_tracklet_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
    final_matching, _, _ = b_matching_via_node_copying(global_tracklet_neuron_graph)
    df_new = combine_tracklets_using_matching(df_tracklets, final_matching)

    # SAVE
    if to_save:
        track_config = project_data.project_config.get_tracking_config()

        output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
        track_config.save_data_in_local_project(df_new, output_df_fname, also_save_csv=True, make_sequential_filename=True)

    return df_new, final_matching, global_tracklet_neuron_graph, worm_obj, all_long_range_matches


def _unpack_for_track_tracklet_matching(project_data):
    track_config = project_data.project_config.get_tracking_config()
    tracklets_and_neurons_class = project_data.tracklets_and_neurons_class
    df_global_tracks = project_data.intermediate_global_tracks
    num_neurons = len(get_names_from_df(df_global_tracks))
    previous_matches = project_data.global2tracklet
    # d_max = track_config.config['final_3d_postprocessing']['max_dist']
    t_template = track_config.config['final_3d_tracks'].get('template_time_point', 1)
    auto_split_conflicts = track_config.config['final_3d_tracks'].get('auto_split_conflicts', True)
    use_multiple_templates = track_config.config['leifer_params']['use_multiple_templates']
    min_overlap = track_config.config['final_3d_postprocessing']['min_overlap_dlc_and_tracklet']
    only_use_previous_matches = track_config.config['final_3d_postprocessing'].get('only_use_previous_matches', False)
    use_previous_matches = track_config.config['final_3d_postprocessing'].get('use_previous_matches', True)
    outlier_threshold = track_config.config['final_3d_postprocessing'].get('outlier_threshold', 1.0)
    min_confidence = track_config.config['final_3d_postprocessing'].get('min_confidence', 0.0)
    tracklet_splitting_iterations = track_config.config['final_3d_postprocessing'].get('tracklet_splitting_iterations', 5)
    return df_global_tracks, min_confidence, min_overlap, num_neurons, only_use_previous_matches, outlier_threshold, \
           previous_matches, t_template, track_config, tracklets_and_neurons_class, use_multiple_templates, \
           use_previous_matches, tracklet_splitting_iterations, auto_split_conflicts


def _save_graphs_and_combined_tracks(df_new, final_matching_no_conflict, final_matching_with_conflict,
                                     global_tracklet_neuron_graph, track_config,
                                     worm_obj, df_tracklets_split):
    track_config.logger.info("Finished calculations, now saving")
    # Save both main products
    output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
    output_df_fname = track_config.save_data_in_local_project(df_new, output_df_fname, also_save_csv=True,
                                                              make_sequential_filename=True)
    output_fname = track_config.config['global2tracklet_matches_fname']
    global2tracklet = final_matching_no_conflict.get_mapping_0_to_1(unique=False)
    output_fname = track_config.pickle_data_in_local_project(global2tracklet, output_fname,
                                                             make_sequential_filename=True)
    updates = {}
    if df_tracklets_split is not None:
        track_config.logger.info("Also saving automatically split tracklets")
        split_df_fname = os.path.join('3-tracking', 'all_tracklets_after_conflict_splitting.pickle')
        track_config.pickle_data_in_local_project(df_tracklets_split, relative_path=split_df_fname,
                                                  custom_writer=pd.to_pickle)
        # TODO: update the name of wiggle_split_tracklets_df_fname
        updates['wiggle_split_tracklets_df_fname'] = split_df_fname

    # Update config file
    output_df_fname = track_config.unresolve_absolute_path(output_df_fname)
    output_fname = track_config.unresolve_absolute_path(output_fname)
    updates.update({'final_3d_tracks_df': str(output_df_fname),
                    'global2tracklet_matches_fname': str(output_fname)})
    track_config.config.update(updates)
    track_config.update_self_on_disk()

    # Additional helper files
    track_config.logger.info("Also saving raw intermediate products")
    dir_name = Path(os.path.join('3-tracking', 'raw'))
    dir_name.mkdir(exist_ok=True)
    if global_tracklet_neuron_graph is not None:
        fname = str(dir_name.joinpath('global_tracklet_neuron_graph.pickle'))
        track_config.pickle_data_in_local_project(global_tracklet_neuron_graph, fname)
    fname = str(dir_name.joinpath('final_matching.pickle'))
    track_config.pickle_data_in_local_project(final_matching_no_conflict, fname)
    if final_matching_with_conflict is not None:
        fname = str(dir_name.joinpath('final_matching_with_conflict.pickle'))
        track_config.pickle_data_in_local_project(final_matching_with_conflict, fname)
    # Sometimes this object is too big, so change the protocol
    fname = str(dir_name.joinpath('worm_obj.pickle'))
    track_config.pickle_data_in_local_project(worm_obj, fname, protocol=4)


def extend_tracks_using_global_tracking(df_global_tracks, df_tracklets, worm_obj: TrackedWorm,
                                        min_overlap=5, min_confidence=0.2, outlier_threshold=1.0,
                                        used_names=None, to_reserve_initialization_tracklets=False,
                                        verbose=0, DEBUG=False):
    """
    For each neuron, get the relevant global track
    Then calculate all track-tracklet distances, using percent inliers
    If passes threshold 1:
      Then check z/volume threshold 2:
        Directly add the tracklets to the neurons within worm_obj
      Else simply do not add

    After this function, do b_matching

    Parameters
    ----------
    project_data
    min_overlap
    min_confidence
    verbose

    Returns
    -------
    No return value; neurons contain all information

    """
    # Pre-make coordinates so that the dataframe is not continuously indexed
    if used_names is None:
        used_names = set()
    coords = ['z', 'x', 'y']
    all_tracklet_names = get_names_from_df(df_tracklets)
    if DEBUG:
        all_tracklet_names = all_tracklet_names[:1000]
    # list_tracklets_zxy = [df_tracklets[name][coords].to_numpy() for name in tqdm(all_tracklet_names)]
    # These dictionaries remove tracklets that are shorter than min_overlap (don't need to check individual tracks)
    # TODO: when tracklets are the length of the entire video, the lengths desynchronize
    dict_tracklets_zxy_small, dict_tracklets_zxy_ind = precalculate_lists_from_dataframe(
        all_tracklet_names, coords, df_tracklets, min_overlap)
    remaining_tracklet_names = list(dict_tracklets_zxy_small.keys())
    if verbose >= 1:
        print(f"Found {len(remaining_tracklet_names)} candidate tracklets")

    _name = remaining_tracklet_names[0]
    logging.info(f"Found tracks of shape: {df_global_tracks.shape}")  # and "
                 # f"{dict_tracklets_zxy_small[_name].shape}")
    if df_global_tracks.shape[0] - 1 == dict_tracklets_zxy_small[_name].shape[0]:
        to_shorten = True
        logging.warning("Tracks are 1 time point shorter than tracklets; removing the last point")
    else:
        to_shorten = False

    # Reserve any tracklets the neurons were initialized with (i.e. the training data)
    if to_reserve_initialization_tracklets:
        for _, neuron in worm_obj.global_name_to_neuron.items():
            used_names.update(neuron.get_raw_tracklet_names())

    # Add new tracklets
    all_conf_output = {}
    for i, (neuron_name, neuron) in enumerate(tqdm(worm_obj.global_name_to_neuron.items(), total=worm_obj.num_neurons)):

        if verbose >= 2:
            print(f"Checking global track {neuron_name}")
        # New: use the track as produced by the global tracking
        # Is currently working: Confirm that the worm_obj has the same neuron names as leifer
        if to_shorten:
            this_global_track = df_global_tracks[neuron_name][coords][:-1]
        else:
            this_global_track = df_global_tracks[neuron_name][coords]
        this_global_track = this_global_track.replace(0.0, np.nan).to_numpy(float)

        dist = calc_global_track_to_tracklet_distances_subarray(this_global_track,
                                                                dict_tracklets_zxy_small, dict_tracklets_zxy_ind,
                                                                min_overlap=min_overlap)

        # Loop through candidates, and attempt to add
        all_summarized_conf = summarize_confidences_outlier_percent(dist, outlier_threshold=outlier_threshold)
        i_sorted_by_confidence = np.argsort(-all_summarized_conf)  # Reverse sort, but keep nans at the end
        all_conf_output[neuron_name] = all_summarized_conf

        num_candidate_neurons = 0
        for num_candidate_neurons, i_tracklet in enumerate(i_sorted_by_confidence):
            # Check if this was used before; note that conflicts may be desired for postprocessing
            candidate_tracklet_name = remaining_tracklet_names[i_tracklet]
            if candidate_tracklet_name in used_names:
                if verbose >= 3:
                    print(f"High confidence, but already used: {candidate_tracklet_name}")
                continue
            # Check distance; break because they are sorted by distance
            this_confidence = all_summarized_conf[i_tracklet]
            if this_confidence <= min_confidence or np.isnan(this_confidence):
                if verbose >= 3:
                    print(f"Confidence {this_confidence} too low: {candidate_tracklet_name}; "
                          f"not checking any further tracklets")
                break

            candidate_tracklet = df_tracklets[[candidate_tracklet_name]]
            is_match_added = neuron.add_tracklet(this_confidence, candidate_tracklet, metadata=candidate_tracklet_name,
                                                 check_using_classifier=False, verbose=verbose-2)

            if verbose >= 3:
                print(f"Tracklet successfully added: {candidate_tracklet_name}")

        if verbose >= 2:
            print(f"{num_candidate_neurons+1} candidate tracklets checked")
            print(f"Tracklets added to make neuron: {neuron}")

        if DEBUG and i > 1:
            break

    return all_conf_output


def extend_tracks_using_similar_postures(all_frames, frame_pair_options, reference_posture,
                                         verbose, worm_obj):
    all_long_range_matches = {}
    anchor_ind = 0
    anchor_frame = all_frames[anchor_ind]

    # Loop over similar postures, use the WormReferencePosture to attempt a long-distance match
    # Then use the neuron match to match tracklets
    indices_to_check = reference_posture.indices_close_to_reference[1:]
    indices_to_check = [i for i in indices_to_check if i < len(all_frames)]
    for i_next_similar_posture in tqdm(indices_to_check):

        # Just always loop through all tracks, even if they (theoretically) don't have a gap
        # tracks_with_gap = worm_obj.tracks_with_gap_at_or_after_time(i_next_similar_posture)
        # if not tracks_with_gap:
        #     continue

        # Then do one volume-volume match to try and continue all ended tracklets
        pair_indices = (anchor_ind, i_next_similar_posture)
        long_range_pair = all_long_range_matches.get(pair_indices, None)
        if long_range_pair is None:
            long_range_frame = all_frames[i_next_similar_posture]
            if verbose >= 2:
                print(f"Calculating matches for pair: {pair_indices}")
            long_range_pair = calc_FramePair_from_Frames(frame0=anchor_frame, frame1=long_range_frame,
                                                         frame_pair_options=frame_pair_options)

            all_long_range_matches[pair_indices] = long_range_pair
        else:
            if verbose >= 3:
                print(f"Reusing matches for pair: {pair_indices}")

        # Build convenience class
        long_range_pair.calc_final_matches()
        long_range_matches = MatchesWithConfidence.matches_from_array(np.array(long_range_pair.final_matches))
        mapping_to_long_range = long_range_matches.get_mapping_0_to_1()
        mapping_to_confidence = long_range_matches.get_mapping_pair_to_conf()

        tracks_that_are_filled = 0
        for track_name, track in tqdm(worm_obj.global_name_to_neuron.items(), total=worm_obj.num_neurons, leave=False):
        # for track_name, track in tracks_with_gap.items():

            # From the starting neuron, get the long-range match
            i_starting_neuron = track.neuron_ind
            i_matched_neuron = mapping_to_long_range.get(i_starting_neuron, None)
            if i_matched_neuron is None:
                continue
            conf = mapping_to_confidence[(i_starting_neuron, i_matched_neuron)]

            # From the long-range match (including frame information), get the tracklet
            # For now, just accept it
            _, matched_tracklet_name = worm_obj.detections.get_tracklet_from_neuron_and_time(i_matched_neuron,
                                                                                             i_next_similar_posture)
            if matched_tracklet_name is None:
                # i.e. there was a neuron match, but it doesn't belong to any tracklet
                continue
            matched_tracklet_df = worm_obj.detections.df_tracklets_zxy[[matched_tracklet_name]]
            track.add_tracklet(confidence=conf,
                               tracklet=matched_tracklet_df,
                               metadata=f"Match due to pair {pair_indices}; original name {matched_tracklet_name}")

            # Enhancement: Also record if the matched neurons match the tracklets that didn't end
            tracks_that_are_filled += 1

        if verbose >= 2:
            print(f"At time {i_next_similar_posture}, extended {tracks_that_are_filled} tracks")

    return all_long_range_matches


def initialize_worm_object(df_tracklets, segmentation_metadata):
    detections = DetectedTrackletsAndNeurons(df_tracklets,
                                             segmentation_metadata)
    worm_obj = TrackedWorm(detections=detections, verbose=1)
    worm_obj.initialize_neurons_at_time(t=0)
    return worm_obj


def combine_tracklets_using_matching(df_tracklets, final_matching):
    # Finally, make the full dataframe
    neuron2tracklet_dict = final_matching.get_mapping_0_to_1(unique=False)
    neuron_names = list(neuron2tracklet_dict.keys())
    df_new = empty_dataframe_like(df_tracklets, neuron_names)

    max_t = len(df_tracklets)
    id_vector = np.zeros(max_t)
    # Actually join
    logging.info(f"Combining tracklets into full dataframe with {len(neuron_names)} neurons")
    for neuron_name, tracklet_list in tqdm(neuron2tracklet_dict.items()):
        for tracklet_name in tracklet_list:
            this_tracklet = df_tracklets[tracklet_name]
            # Preprocess the tracklet dataframe to have an additional column: the id of the tracklet

            nonzero_ind = this_tracklet['z'].notnull()
            tracklet_id = int(tracklet_name.split('_')[-1])

            df_new[neuron_name] = df_new[neuron_name].combine_first(this_tracklet)
            try:
                df_new.loc[nonzero_ind, (neuron_name, 'raw_tracklet_id')] = tracklet_id
            except KeyError:
                id_vector[:] = np.nan
                id_vector[nonzero_ind] = tracklet_id
                df_new[neuron_name, 'raw_tracklet_id'] = id_vector

    if hasattr(df_new, 'sparse'):
        df_new = df_new.sparse.to_dense()
    return df_new


def bipartite_matching_on_each_time_slice(global_tracklet_neuron_graph, df_tracklets) -> MatchesWithConfidence:
    """
    As an alternative to b_matching_via_node_copying, do a separate bipartite matching problem on the small subgraphs of
    tracklets defined for each time point

    Note that this nearly but not completely enforces time-uniqueness
    Specifically, if there is a hierarchy within 3 tracklets and two neurons, such that at one time tracklet_001
    outcompetes other matches, but after tracklet_001 ends tracklet_002 becomes the best match, AND tracklet_001 and
    tracklet_002 overlap in time

    Returns
    -------

    """
    bipartite_slice_matches = MatchesWithConfidence()
    neuron_nodes = global_tracklet_neuron_graph.get_nodes_of_class(0)

    print("Precalculating tracklets that exist at each time point")
    time2names_mapping = defaultdict(list)
    names = get_names_from_df(df_tracklets)
    for name in tqdm(names):
        ind = df_tracklets.loc[:, (name, 'z')].dropna(axis=0).index
        for t in ind:
            time2names_mapping[int(t)].append(name)

    print("Matching the tracks to each tracklet")
    for t, names_at_time in tqdm(time2names_mapping.items()):
        names_at_time.sort()

        # Convert raw tracklet names to node names
        tracklet_names = [global_tracklet_neuron_graph.raw_name_to_network_name(n) for n in names_at_time]

        # Also add the neurons
        network_names_at_time = tracklet_names.copy()
        network_names_at_time.extend(neuron_nodes)

        # Build subgraph and do matching
        subgraph = global_tracklet_neuron_graph.subgraph(network_names_at_time).copy()
        subgraph_matching = nx.bipartite.maximum_matching(subgraph, top_nodes=neuron_nodes)

        # Get confidence, and add them back to the new object
        for k, v in subgraph_matching.items():
            # This dict has matches in both directions
            if k not in neuron_nodes:
                continue

            k_raw = subgraph.network_name_to_raw_name(k)
            v_raw = subgraph.network_name_to_raw_name(v)
            conf = subgraph.get_edge_data(k, v)['weight']
            new_match = [k_raw, v_raw, conf]

            if not bipartite_slice_matches.match_already_exists(new_match):
                reason = f"Time={t}"
                bipartite_slice_matches.add_match(new_match, reason)

    return bipartite_slice_matches


def greedy_matching_using_node_class(no_conflict_neuron_graph, node_class_to_match=1):
    """
    Greedily matches one class of nodes.

    Assumes that the greedy match must win every competition it is a part of, not just one
    """
    tracklet_nodes = no_conflict_neuron_graph.get_nodes_of_class(node_class_to_match)
    final_matching = MatchesWithConfidence()
    for tracklet_name in list(tracklet_nodes):
        these_matches = dict(no_conflict_neuron_graph[tracklet_name])

        # Greedy matching
        weights = [val['weight'] for val in these_matches.values()]
        i_max = np.argmax(weights)
        neuron_name = list(these_matches.keys())[i_max]

        new_match = [neuron_name, tracklet_name, weights[i_max]]
        final_matching.add_match(new_match)
    return final_matching


def b_matching_via_node_copying(global_tracklet_neuron_graph):
    """
    NOTE: there is apparently a bug that causes incorrect matching when there are too many copies...

    Solves the many-to-one version of the bipartite matching problem

    Copy the bipartite=0 nodes so that this becomes a normal bipartite matching problem,
    i.e. one-to-one, not many-to-one

    In more detail:
    1. Get the nodes of class 0 (neurons) and 1 (tracklets)
        - Final matching will be 100 tracklets per neuron (~150 neurons total)
        - Currently they are overmatched, and have >1000 tracklet candidates per neuron
    2. For each candidate tracklet, create a copy of the neuron node and add to a new graph
        - Copies all of the edges as well, so that there are now 1000 tracklets matched to 1 neuron and 999 copies
        - Now there are many more neurons than tracklets
    3. Do normal bipartite matching
        - Note: must explicitly decide which nodes to get matches, and it shouldn't be the copy nodes
    4. Collapse the expanded neuron copies back to their original names

    NOTE: assumes the 'metadata' field of the nodes should be used as the final match names
    """
    raise NotImplementedError
    logging.info("Matching using b_matching_via_node_copying")
    neuron_nodes = global_tracklet_neuron_graph.get_nodes_of_class(0)
    tracklet_nodes = global_tracklet_neuron_graph.get_nodes_of_class(1)
    global_tracklet_neuron_graph_with_copies = global_tracklet_neuron_graph.subgraph(tracklet_nodes).copy()
    new_name_to_original_name = dict()
    for n in neuron_nodes:
        original_edges = list(global_tracklet_neuron_graph.edges(n, data=True))

        for i_copy in range(len(original_edges)):
            # num_copies_already_added = len(original_neuron_to_copy_mapping[n])
            new_name = f"{n}_copy{i_copy}"
            new_name_to_original_name[new_name] = n

            # Add a copy of the node for each edge, which has all the original edges
            new_edges = [[new_name, e[1], e[2]] for e in original_edges]
            global_tracklet_neuron_graph_with_copies.add_edges_from(new_edges)

    # Do normal bipartite matching, such that each tracklet gets a match to some copy of a neuron
    # Note: the neuron copies don't have metadata set
    g = global_tracklet_neuron_graph_with_copies
    extended_tracklet_nodes = {n for n, d in g.nodes(data=True) if "tracklet" in n}
    matching_with_copies = nx.bipartite.maximum_matching(g, top_nodes=extended_tracklet_nodes)

    # Collapse the added copies back to the original neuron, to get a many-to-one matching
    final_matching = MatchesWithConfidence()
    for name0, name1 in matching_with_copies.items():
        if 'neuron' in name0:
            neuron_copy_name = name0
            tracklet_raw_name = name1
        else:
            continue
            # tracklet_raw_name = name0
            # neuron_copy_name = name1

        # Get the names as they are in the graph above
        # Note that all the copies of the neurons do NOT have the same weights to a given tracklet
        neuron_raw_name = new_name_to_original_name[neuron_copy_name]
        weight = global_tracklet_neuron_graph[neuron_raw_name][tracklet_raw_name]['weight']

        # Get the final names, as they are in the previous dataframes
        tracklet_name = global_tracklet_neuron_graph.nodes[tracklet_raw_name]['metadata']
        neuron_name = global_tracklet_neuron_graph.nodes[neuron_raw_name]['metadata']

        new_match = [neuron_name, tracklet_name, weight]

        final_matching.add_match(new_match)
    return final_matching, matching_with_copies, global_tracklet_neuron_graph_with_copies
