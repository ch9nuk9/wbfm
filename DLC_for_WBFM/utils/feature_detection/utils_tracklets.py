import pandas as pd
import numpy as np
from tqdm import tqdm
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import calc_2frame_matches_using_class

##
## Helper functions for tracks
##

def create_new_track(i0, i1,
                    i0_xyz,
                    i1_xyz,
                    i1_prob,
                    next_clust_ind,
                    this_point_cloud_offset,
                    next_point_cloud_offset,
                    which_slice,
                    i1_global,
                    verbose=0):

    new_track = pd.DataFrame({'clust_ind':next_clust_ind,
                              'all_ind_local':[[i0,i1]],
                              'all_ind_global':[[this_point_cloud_offset+i0, i1_global]],
                              'all_xyz':[[i0_xyz,i1_xyz]],
                              'all_prob':[[i1_prob]],
                              'slice_ind':[[which_slice, which_slice+1]],
                              'extended_this_slice':True,
                              'not_finished':True})

    if verbose >= 1:
        print(f"Creating new track {next_clust_ind}")
    next_clust_ind = next_clust_ind + 1

    return next_clust_ind, new_track


def append_to_track(row, i, clust_df, which_slice, i1, i1_global,
                 i1_xyz, i1_prob,
                 append_or_extend=list.append,
                 verbose=0):
    """
    Parameters
    ===============
    row : dataframe
        As the full dataframe is iterated through, this is the row to be extended
    i : int
        The index of the full dataframe row that is being appended
    clust_df : dataframe
        The full dataframe. The 'i'th row should be 'row' (above)
    which_slice : int
        The slice (frame) corresponding to the index 'i'
    i1 : int
        The local index of the matched neuron
    i1_global : int
        The global coordinate representation of 'i1'
    i1_xyz : tuple(x,y,z)
        The xyz coordinates of the neuron 'i1'
    i1_prob : float
        The probability of the neuron being added to this tracklet
        For now, determined by simple sequential matching
    append_or_extend : function
        list.append or list.extend
        Usage:
            list.append is used when adding a single neuron to the row
            list.extend allows full rows to be combined

    """
    append_or_extend(row['all_ind_local'], i1)
    clust_df.at[i,'all_ind_local'] = list(row['all_ind_local'])
    append_or_extend(row['all_ind_global'], i1_global)
    clust_df.at[i,'all_ind_global'] = list(row['all_ind_global'])
    if type(i1_xyz) is np.ndarray:
        np.vstack([row['all_xyz'], i1_xyz])
        clust_df.at[i,'all_xyz'] = row['all_xyz']
    else:
        append_or_extend(row['all_xyz'], i1_xyz)
        clust_df.at[i,'all_xyz'] = list(row['all_xyz'])
    append_or_extend(row['all_prob'], i1_prob)
    clust_df.at[i,'all_prob'] = list(row['all_prob'])
    if type(which_slice) is int:
        append_or_extend(row['slice_ind'], which_slice+1)
    else:
        # Assume they are already the to-be-matched indices
        append_or_extend(row['slice_ind'], which_slice)
    clust_df.at[i,'slice_ind'] = list(row['slice_ind'])
    clust_df.at[i,'extended_this_slice'] = True

    if verbose >= 1:
        clust_ind = row['clust_ind']
        tmp = len(row['all_ind_local'])
        print(f"Adding to track {clust_ind} (length {tmp}) on slice {which_slice}")

    return clust_df


##
## Main function
##

def build_tracklets_from_matches(all_neurons,
                                 all_matches,
                                 all_likelihoods=None,
                                 verbose=0):
    """
    Builds tracklets from an array of pairwise matches

    Can accept points as list of np.arrays
    Can accept matchings as list of np.arrays
    """

    if all_likelihoods is None:
        all_likelihoods = [np.ones(len(m)) for m in all_matches]

    # Use registration results to build a combined and colored pointcloud
    columns=['clust_ind', 'all_ind_local', 'all_ind_global', 'all_xyz',
             'all_prob',
             'slice_ind','extended_this_slice', 'not_finished']
    clust_df = pd.DataFrame(columns=columns)

    next_clust_ind = 0
    this_point_cloud_offset = 0
    next_point_cloud_offset = 0

    for i_match, match in tqdm(enumerate(all_matches), total=len(all_matches)):
        if verbose >= 1:
            print("==============================================================")
            print(f"{i_match} / {len(all_matches)}")
        # Get transform to global coordinates
        these_neurons = all_neurons[i_match]
        this_xyz = np.asarray(these_neurons)
        next_neurons = all_neurons[i_match+1]
        next_xyz = np.asarray(next_neurons)
        next_point_cloud_offset = next_point_cloud_offset + len(these_neurons)

        # Get the probabilities as well
        this_prob = all_likelihoods[i_match]
        this_prob = np.asarray(this_prob)
        # next_prob = all_likelihoods[i_match+1]
        # next_prob = np.asarray(next_prob)

        offsets = {'next_point_cloud_offset':next_point_cloud_offset,
                   'this_point_cloud_offset':this_point_cloud_offset,
                   'which_slice':i_match}

        pairs = np.asarray(match)
        # Initialize ALL as to-be-finished
        clust_df['extended_this_slice'] = False

        all_new_tracks = []
        for i_pair, (i0, i1) in enumerate(pairs):
            next_point = {'i1':i1,
                          'i1_xyz':next_xyz[i1,:],
                          'i1_prob':this_prob[i_pair], # Probability is attached to the pair
                          'i1_global':next_point_cloud_offset+i1}
            current_point = {'i0':i0,
                             'i0_xyz':this_xyz[i0,:],
                             'next_clust_ind':next_clust_ind}
            # If no tracks, initialize
            ind_to_check = clust_df['not_finished'] & ~clust_df['extended_this_slice']

            if verbose >= 2:
                print(f"Clusters available to check: {np.where(ind_to_check)}")
            if len(ind_to_check)==0:
                next_clust_ind, new_track = create_new_track(**current_point, **next_point, **offsets)
                all_new_tracks.append(new_track)
                continue
            # Add to previous track if possible
            for i, row in clust_df[ind_to_check].iterrows():
                if verbose >= 2:
                    print(f"pair: {i0}, {i1}, trying cluster: {i}")
                if i0 == row['all_ind_local'][-1]:
                    clust_df = append_to_track(row, i, clust_df, i_match, **next_point)
                    break
            else:
                # Create new track
                next_clust_ind, new_track = create_new_track(**current_point, **next_point, **offsets)
                all_new_tracks.append(new_track)

        # Actually add the tracks to the dataframe
        for t in all_new_tracks:
            clust_df = clust_df.append(t, ignore_index=True)

        # Finalize tracks that didn't get a new point this loop
        to_finish = ~clust_df['extended_this_slice'].astype(bool)
        if len(np.where(to_finish)[0]) > 0 and verbose >= 1:
            print(f"Finished tracks {np.where(to_finish)[0]}")
        clust_df.loc[to_finish,'not_finished'] = False

        if verbose >= 3 and len(pairs) > 0:
            print("WIP")
            #visualize_tracks_simple(this_pc, next_pc, pairs)

        this_point_cloud_offset = next_point_cloud_offset

    clust_df['all_xyz'] = clust_df['all_xyz'].apply(np.array)

    return clust_df


def build_tracklets_from_classes(all_frames,
                                 all_matches_dict,
                                 all_likelihoods_dict=None,
                                 verbose=0):
    """
    Build tracklets starting from a different format

    See also: build_tracklets_from_matches
    """

    if type(all_matches_dict) != dict:
        print("Expected dictionary of pairwise matches")
        print("Did you mean to call build_tracklets_from_matches()?")
        raise ValueError

    # Input of build_tracklets_from_matches:
    #   1. List of all pairwise matches
    #   2. List of all neuron 3d locations

    all_neurons = [all_frames[0].neuron_locs]
    all_matches = []
    if all_likelihoods_dict is None:
        all_likelihoods = None
    else:
        all_likelihoods = []
    for i in range(1,len(all_frames)):
        # Get matches and conf
        key = (i-1,i)
        all_matches.append(all_matches_dict[key])
        if all_likelihoods is not None:
            all_likelihoods.append(all_likelihoods_dict[key])

        all_neurons.append(all_frames[i].neuron_locs)

    # Call old function
    return build_tracklets_from_matches(all_neurons,
                                        all_matches,
                                        all_likelihoods,
                                        verbose=verbose)


##
## Postprocessing: stitching tracklets together
##

def consolidate_tracklets(df, tracklet_matches, verbose=0):
    # Consolidate tracklets using matches
    rows_to_drop = []
    for row0_ind, row1_ind in tracklet_matches:
        base_row = df.loc[row0_ind]
        row_to_add = df.loc[row1_ind]

        df = append_to_track(base_row,
                             row0_ind,
                             df,
                             row_to_add['slice_ind'],
                             row_to_add['all_ind_local'],
                             row_to_add['all_ind_global'],
                             row_to_add['all_xyz'],
                             row_to_add['all_prob'],
                             append_or_extend=list.extend)
        rows_to_drop.append(row1_ind)

    df.drop(rows_to_drop, axis=0)
    if verbose >= 1:
        print(f"Extended and dropped {len(rows_to_drop)}/{df.shape[0]} rows")

    return df


def stitch_tracklets(clust_df,
                     all_frames,
                     max_stitch_distance=10,
                     min_starting_tracklet_length=3,
                     minimum_match_confidence=0.4,
                     verbose=0):
    """
    Takes tracklets in a dataframe and attempts to stitch them together
    Uses list of original frame data

    Only attempts to match the last frame to first frames of other tracklets
    """
    if verbose >= 1:
        print(f"Trying to consolidate {clust_df.shape[0]} tracklets")
        print("Note: computational time of this function is front-loaded")
    # Get tracklet starting and ending indices in frame space
    all_starts = clust_df['slice_ind'].apply(lambda x : x[0])
    all_ends = clust_df['slice_ind'].apply(lambda x : x[-1])
    all_long_enough = np.where(all_ends - all_starts > min_starting_tracklet_length)[0]
    is_available = clust_df['slice_ind'].apply(lambda x : True)

    # Reuse distant matches calculations
    distant_matches_dict = {}
    distant_conf_dict = {}
    tracklet_matches = []
    tracklet_conf = []

    for ind in tqdm(all_long_enough):
        # Get frame and individual neuron to match
        i_end_frame = all_ends.at[ind]
        frame0 = all_frames[i_end_frame]
        neuron0 = clust_df.at[ind,'all_ind_local'][-1]

        # Get all close-by starts
        start_is_after = all_starts.gt(i_end_frame+1)
        start_is_close = all_starts.lt(max_stitch_distance+i_end_frame)
        tmp = start_is_after & start_is_close & is_available
        possible_start_tracks = np.where(tmp)[0]
        if len(possible_start_tracks)==0:
            continue

        # Loop through possible next tracklets
        for i_start_track in possible_start_tracks:
            i_start_frame = all_starts.at[i_start_track]
            frame1 = all_frames[i_start_frame]
            neuron1 = clust_df.at[i_start_track,'all_ind_local'][0]

            if verbose >= 4:
                print(f"Trying to match tracklets {ind} and {i_start_track}")
            key = (i_end_frame, i_start_frame)
            if key in distant_matches_dict:
                matches = distant_matches_dict[key]
                conf = distant_conf_dict[key]
            else:
                # Otherwise, calculate from scratch
                if verbose >= 3:
                    print(f"Calculating new matches between frames {key}")
                out = calc_2frame_matches_using_class(frame0, frame1)
                matches, conf = out[0], out[1]
                # Save for future
                distant_matches_dict[key] = matches
                distant_conf_dict[key] = conf

            # Find if these specific neurons are matched in the frames
            n_key = [neuron0, neuron1]
            t_key = [ind, i_start_track]
            if n_key in matches:
                this_conf = conf[matches.index(n_key)]
                if verbose >= 2:
                    print(f"Matched tracks {t_key} with confidence {this_conf}")
                    print(f"(frames {key} and neurons {n_key})")
                if this_conf < minimum_match_confidence:
                    #2err
                    continue
                tracklet_matches.append(t_key)
                tracklet_conf.append(this_conf)
                is_available.at[i_start_track] = False
                # TODO: just take the first match
                break

    df = consolidate_tracklets(clust_df.copy(), tracklet_matches, verbose)
    if verbose >= 1:
        print("Finished")
    intermediates = (distant_matches_dict, distant_conf_dict, tracklet_matches, all_starts, all_ends)
    return df, intermediates
