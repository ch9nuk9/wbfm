import pandas as pd
import numpy as np

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


def extend_track(row, i, clust_df, which_slice, i1, i1_global,
                 i1_xyz, i1_prob, verbose=0):
    row['all_ind_local'].append(i1)
    clust_df.at[i,'all_ind_local'] = list(row['all_ind_local'])
    row['all_ind_global'].append(i1_global)
    clust_df.at[i,'all_ind_global'] = list(row['all_ind_global'])
    row['all_xyz'].append(i1_xyz)
    clust_df.at[i,'all_xyz'] = list(row['all_xyz'])
    row['all_prob'].append(i1_prob)
    clust_df.at[i,'all_prob'] = list(row['all_prob'])
    row['slice_ind'].append(which_slice+1)
    clust_df.at[i,'slice_ind'] = list(row['slice_ind'])
    clust_df.at[i,'extended_this_slice'] = True

    clust_ind = row['clust_ind']
    tmp = len(row['all_ind_local'])
    if verbose >= 1:
        print(f"Adding to track {clust_ind} (length {tmp}) on slice {which_slice}")

    return clust_df

##
## Main function
##

def build_tracklets_from_matches(all_neurons,
                                 all_matches,
                                 all_likelihoods=None,
                                 verbose = 0):
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
                    clust_df = extend_track(row, i, clust_df, i_match, **next_point)
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
