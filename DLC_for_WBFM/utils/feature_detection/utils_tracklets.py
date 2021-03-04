import pandas as pd
import numpy as np
from tqdm import tqdm
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import ReferenceFrame
import copy
from collections import defaultdict

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
    r = row['all_ind_local'][:]
    append_or_extend(r, i1)
    clust_df.at[i,'all_ind_local'] = r

    r = row['all_ind_global'][:]
    append_or_extend(r, i1_global)
    clust_df.at[i,'all_ind_global'] = r

    r = row['all_xyz'][:]
    if type(i1_xyz) is np.ndarray:
        # row['all_xyz'] = np.vstack([row['all_xyz'], i1_xyz])
        clust_df.at[i,'all_xyz'] = np.vstack([r, i1_xyz])
    else:
        append_or_extend(r, i1_xyz)
        clust_df.at[i,'all_xyz'] = r
    append_or_extend(row['all_prob'], i1_prob)

    r = row['all_prob'][:]
    clust_df.at[i,'all_prob'] = r

    r = row['slice_ind'][:]
    if type(which_slice) is int:
        append_or_extend(r, which_slice+1)
    else:
        # Assume they are already the to-be-matched indices
        append_or_extend(r, which_slice)
    clust_df.at[i,'slice_ind'] = r

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

    Parameters
    ===============
    all_neurons : list
        List of lists, with integers for all neurons present in a frame
    all_matches : list
        Matches between adjacent frames in the coordinates local to each frame
    all_likelihoods : list
        Same as all_matches, but for the confidence of the matchs

    See build_tracklets_from_classes() if using ReferenceFrame class
    """

    if type(all_neurons[0]) == ReferenceFrame:
        print("===============")
        print("Found type ReferenceFrame")
        print("Did you mean to call build_tracklets_from_classes()?")
        raise ValueError

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

    #for i_match, match in tqdm(enumerate(all_matches), total=len(all_matches)):
    for i_match, match in enumerate(all_matches):
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
    # if type(all_frames)==dict:
        # TODO: make the below loops work for dict
        # all_frames = list(all_frames.values())
        # print("If this is a dict, then the indices are probably off.")
        # raise ValueError
    try:
        all_neurons = [all_frames[0].neuron_locs]
        final_frame_ind = len(all_frames)
        start_frame_ind = 0
    except:
        k = list(all_frames)
        all_neurons = [all_frames[k[0]].neuron_locs]
        start_frame_ind = min(k)
        final_frame_ind = max(k)
    all_matches = []
    if all_likelihoods_dict is None:
        all_likelihoods = None
    else:
        all_likelihoods = []
    for i in range(1, final_frame_ind):
        # Pad the initials with empties if this is a dict
    # for key, i in zip(all_matches_dict, all_frames):
        # Get matches and conf
        key = (i-1,i)
        if key in all_matches_dict:
            all_matches.append(all_matches_dict[key])
        else:
            all_matches.append([])
        if all_likelihoods is not None:
            all_likelihoods.append(all_likelihoods_dict[key])

        if i < start_frame_ind:
            all_neurons.append([])
        else:
            all_neurons.append(all_frames[i].neuron_locs)
        # all_neurons.append(frame.neuron_locs)

    # Call old function
    return build_tracklets_from_matches(all_neurons,
                                        all_matches,
                                        all_likelihoods,
                                        verbose=verbose)



def build_tracklets_simple(all_matches, verbose=0):
    """
    Build tracklets without requiring other data
        Especially neuron position

    See also: build_tracklets_from_matches
    """

    num_neurons = 0
    for m in all_matches:
        max_neuron = np.max(np.array(m))
        num_neurons = max(num_neurons, max_neuron)

    def dummy_xyz_factory(num_neurons=num_neurons):
        return np.zeros((num_neurons,3))

    all_neurons = defaultdict(dummy_xyz_factory)

    # Call old function
    return build_tracklets_from_matches(all_neurons,
                                        all_matches,
                                        None,
                                        verbose=verbose)

##
## Postprocessing: stitching tracklets together
##

def consolidate_tracklets(df_raw, tracklet_matches, verbose=0):
    """Consolidate tracklets using matches

    Note: assumes that the indices in tracklet_matches correspond to the clust_ind column
    """
    base_of_dropped_rows = {}
    rows_to_drop = []
    df = copy.deepcopy(df_raw)
    for row0_ind, row1_ind in tracklet_matches:
        # If we have two matches: (0,1) and later (1,10), add directly to track 0
        # TODO: what if the matches are out of order?
        if row0_ind in rows_to_drop:
            row0_ind = base_of_dropped_rows[row0_ind]
        base_row = df.loc[row0_ind].copy(deep=True)
        row_to_add = df.loc[row1_ind].copy(deep=True)
        if verbose >= 2:
            print(f"Adding track {row1_ind} to track {row0_ind}")


        df = append_to_track(base_row,
                             row0_ind,
                             df,
                             row_to_add['slice_ind'][:],
                             row_to_add['all_ind_local'][:],
                             row_to_add['all_ind_global'][:],
                             row_to_add['all_xyz'][:],
                             row_to_add['all_prob'][:],
                             append_or_extend=list.extend)
        base_of_dropped_rows[row1_ind] = row0_ind
        rows_to_drop.append(row1_ind)

    if verbose >= 1:
        print(f"Extended and dropped {len(rows_to_drop)}/{df.shape[0]} rows")

    return df.drop(rows_to_drop, axis=0)


##
## API functions
##

def convert_from_dict_to_lists(tmp_matches, tmp_conf, tmp_neurons):
    # Convert from dict and Frame objects to just lists
    all_matches, all_conf = [], []
    all_neurons = []
    for i in range(len(tmp_matches)-1):
        # Assume keys describe pairwise matches
        k = (i, i+1)
        all_matches.append(tmp_matches[k])
        all_conf.append(tmp_conf[k])
        # This is a ReferenceFrame object
        all_neurons.append(tmp_neurons[i].neuron_locs)
    # This list is one element longer
    all_neurons.append(tmp_neurons[-1].neuron_locs)

    return all_matches, all_conf, all_neurons
