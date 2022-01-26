import logging

import numpy as np

from DLC_for_WBFM.utils.external.utils_networkx import dist2conf
from DLC_for_WBFM.utils.feature_detection.class_frame_pair import calc_FramePair_from_Frames
from DLC_for_WBFM.utils.pipeline.matches_class import MatchesWithConfidence
from DLC_for_WBFM.utils.pipeline.tracklet_class import DetectedTrackletsAndNeurons, TrackedWorm
from DLC_for_WBFM.utils.postprocessing.combine_tracklets_and_DLC_tracks import calc_global_track_to_tracklet_distances
from DLC_for_WBFM.utils.postures.centerline_pca import WormFullVideoPosture, WormReferencePosture
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
import networkx as nx
from networkx.algorithms import bipartite
from collections import defaultdict

from tqdm.auto import tqdm


def long_range_matches_from_config(project_path, to_save=True, verbose=2):
    # project_path = "/home/charles/dlc_stacks/worm3-newseg-2021_11_17/project_config.yaml"

    project_data = ProjectData.load_final_project_data_from_config(project_path, to_load_tracklets=True, to_load_frames=True)
    df_tracklets = project_data.df_all_tracklets
    segmentation_metadata = project_data.segmentation_metadata
    all_frames = project_data.raw_frames
    all_matches = project_data.raw_matches
    raw_clust = project_data.raw_clust

    frame_pair_options = all_matches[(0, 1)].options

    # TODO: move the centerlines to config files
    fname = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv"
    fname_X = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_X_coords.csv"
    fname_Y = "/project/neurobiology/zimmer/wbfm/centerline/wbfm_ulises_centerline_for_charlie/2021-03-04_16-17-30_worm3_ZIM2051-_spline_Y_coords.csv"

    full_posture = WormFullVideoPosture(fname, fname_X, fname_Y)
    reference_posture = WormReferencePosture(0, full_posture)

    # Initialize TrackedNeurons at 0, and initialize the TrackedWorm
    # Get all tracklets that start at t=0
    all_tracklet_names = df_tracklets.columns.get_level_values(0).drop_duplicates()

    worm_obj = initialize_worm_object(df_tracklets, raw_clust, segmentation_metadata)

    all_long_range_matches = extend_tracks_using_similar_postures(all_frames, frame_pair_options,
                                                                  reference_posture, verbose, worm_obj)

    global_tracklet_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
    final_matching = b_matching_via_node_copying(global_tracklet_neuron_graph)
    df_new = combine_tracklets_using_matching(all_tracklet_names, df_tracklets, final_matching)

    # SAVE
    if to_save:
        track_config = project_data.project_config.get_tracking_config()

        output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
        track_config.h5_in_local_project(df_new, output_df_fname, also_save_csv=True, make_sequential_filename=True)

    return df_new, final_matching, global_tracklet_neuron_graph, worm_obj, all_long_range_matches


def global_track_matches_from_config(project_path, to_save=True, verbose=2, DEBUG=False):
    # Initialize project data and unpack
    project_data = ProjectData.load_final_project_data_from_config(project_path, to_load_tracklets=True)
    df_tracklets = project_data.df_all_tracklets
    tracklets_and_neurons_class = project_data.tracklets_and_neurons_class
    df_global_tracks = project_data.intermediate_global_tracks
    df_training_data = project_data.df_training_tracklets

    all_tracklet_names = df_tracklets.columns.get_level_values(0).drop_duplicates()

    worm_obj = TrackedWorm(detections=tracklets_and_neurons_class, verbose=verbose)
    worm_obj.initialize_neurons_from_training_data(df_training_data)
    worm_obj.initialize_all_neuron_tracklet_classifiers()
    if verbose >= 1:
        print(f"Initialized worm object: {worm_obj}")

    # Add all candidates to neurons
    extend_tracks_using_global_tracking(df_global_tracks, df_tracklets, worm_obj,
                                        min_overlap=5, d_max=5, verbose=verbose, DEBUG=DEBUG)

    # Create
    global_tracklet_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
    final_matching = b_matching_via_node_copying(global_tracklet_neuron_graph)
    df_new = combine_tracklets_using_matching(all_tracklet_names, df_tracklets, final_matching,
                                              num_neurons=worm_obj.num_neurons)

    # SAVE
    if to_save:
        track_config = project_data.project_config.get_tracking_config()

        output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']
        track_config.h5_in_local_project(df_new, output_df_fname, also_save_csv=True, make_sequential_filename=True)

        updates = {'final_3d_tracks_df': str(output_df_fname)}
        track_config.config.update(updates)
        track_config.update_on_disk()

        output_fname = track_config.config['global2tracklet_matches_fname']
        global2tracklet = final_matching.get_mapping_0_to_1(unique=False)
        track_config.pickle_in_local_project(global2tracklet, output_fname, make_sequential_filename=True)

        # Also save raw intermediate products
        fname = '3-tracking/global_tracklet_neuron_graph.pickle'
        track_config.pickle_in_local_project(global_tracklet_neuron_graph, fname)
        fname = '3-tracking/worm_obj.pickle'
        track_config.pickle_in_local_project(worm_obj, fname)
        fname = '3-tracking/final_matching.pickle'
        track_config.pickle_in_local_project(final_matching, fname)

    return df_new, final_matching, global_tracklet_neuron_graph, worm_obj


def extend_tracks_using_global_tracking(df_global_tracks, df_tracklets, worm_obj: TrackedWorm,
                                        min_overlap=5, d_max=5, verbose=0, DEBUG=False):
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
    d_max
    verbose

    Returns
    -------
    No return value; neurons contain all information

    """
    # Pre-make coordinates so that the dataframe is not continuously indexed
    coords = ['z', 'x', 'y']
    all_tracklet_names = list(df_tracklets.columns.levels[0])
    list_tracklets_zxy = [df_tracklets[name][coords].to_numpy() for name in tqdm(all_tracklet_names)]

    # Reserve any tracklets the neurons were initialized with (i.e. the training data)
    used_names = set()
    for _, neuron in worm_obj.global_name_to_neuron.items():
        used_names.update(neuron.get_raw_tracklet_names())

    # Add new tracklets
    for i, (name, neuron) in enumerate(tqdm(worm_obj.global_name_to_neuron.items())):

        # New: use the track as produced by the global tracking
        # TODO: confirm that the worm_obj has the same neuron names as leifer
        this_global_track = df_global_tracks[name][coords][:-1].replace(0.0, np.nan).to_numpy(float)

        # TODO: calculate distance using percent inliers
        dist = calc_global_track_to_tracklet_distances(this_global_track, list_tracklets_zxy,
                                                       min_overlap=min_overlap)

        # Loop through candidates, and attempt to add
        all_summarized_dist = list(map(lambda x: np.nanquantile(x, 0.1), dist))
        i_sorted_by_median_distance = np.argsort(all_summarized_dist)
        num_candidate_neurons = 0
        for num_candidate_neurons, i_tracklet in enumerate(i_sorted_by_median_distance):
            # Check if this was used before
            candidate_name = all_tracklet_names[i_tracklet]
            if candidate_name in used_names:
                continue
            # Check distance; break because they are sorted by distance
            this_distance = all_summarized_dist[i_tracklet]
            if this_distance > d_max:
                break

            candidate_tracklet = df_tracklets[[candidate_name]]
            conf = dist2conf(this_distance)
            is_match_added = neuron.add_tracklet(i_tracklet, conf, candidate_tracklet, metadata=candidate_name,
                                                 check_using_classifier=True)

        if verbose >= 2:
            print(f"{num_candidate_neurons} candidate tracklets")
            print(f"Tracklets added to make neuron: {neuron}")

        if DEBUG and i > 2:
            break


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
            matched_tracklet_ind, matched_tracklet_name = worm_obj.detections.get_tracklet_from_neuron_and_time(i_matched_neuron,
                                                                                                       i_next_similar_posture)
            if matched_tracklet_name is None:
                # i.e. there was a neuron match, but it doesn't belong to any tracklet
                continue
            matched_tracklet_df = worm_obj.detections.df_tracklets_zxy[[matched_tracklet_name]]
            track.add_tracklet(matched_tracklet_ind,
                               confidence=conf,
                               tracklet=matched_tracklet_df,
                               metadata=f"Match due to pair {pair_indices}; original name {matched_tracklet_name}")

            # TODO: Also record if the matched neurons match the tracklets that didn't end
            tracks_that_are_filled += 1

        if verbose >= 2:
            print(f"At time {i_next_similar_posture}, extended {tracks_that_are_filled} tracks")

    return all_long_range_matches


def initialize_worm_object(df_tracklets, raw_clust, segmentation_metadata):
    detections = DetectedTrackletsAndNeurons(df_tracklets,
                                             segmentation_metadata,
                                             df_tracklet_matches=raw_clust)
    worm_obj = TrackedWorm(detections=detections, verbose=1)
    worm_obj.initialize_neurons_at_time(t=0)
    return worm_obj


def combine_tracklets_using_matching(all_tracklet_names, df_tracklets, final_matching, num_neurons):
    # Finally, make the full dataframe
    # Initialize using the index and column structure of the tracklets
    # TODO: Add a column for tracklet ID
    tmp_names = all_tracklet_names[:num_neurons]
    df_new = df_tracklets.loc[:, tmp_names].copy()
    neuron_names = list(set(final_matching.indices0))
    neuron_names.sort()
    name_mapper = {t: n for t, n in zip(tmp_names, neuron_names)}
    df_new.rename(columns=name_mapper, inplace=True)
    df_new[:] = np.nan

    max_t = len(df_tracklets)
    id_vector = np.zeros(max_t)
    # Actually join
    for tracklet_name, neuron_name in tqdm(final_matching.get_mapping_1_to_0().items()):
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

    return df_new


def b_matching_via_node_copying(global_tracklet_neuron_graph):
    """
    Copy the bipartite=0 nodes so that this becomes a normal bipartite matching problem,
    i.e. one-to-one, not many-to-one

    NOTE: assumes the 'metadata' field of the nodes should be used as the final match names
    """
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
    return final_matching
