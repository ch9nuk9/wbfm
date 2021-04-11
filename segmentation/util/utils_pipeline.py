import segmentation.util.utils_postprocessing as post
import numpy as np



def perform_post_processing_2d(mask_array, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3, verbose=0):
    """
    Performs some post-processing steps including: Splitting long neurons, removing short neurons and
    removing too large areas

    Parameters
    ----------
    mask_array : 3D numpy array
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    border_width_to_remove : int
        within that distance to border, artefacts/masks will be removed
    to_remove_border : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    final_masks : 3D numpy array
        3D array of masks after post-processing

    """
    if verbose >= 1:
        print(f"Starting preprocessing with {len(np.unique(mask_array))-1} neurons")
        print("Note: not yet stitched in z")
    masks = post.remove_large_areas(mask_array, verbose=verbose)
    if verbose >= 1:
        print(f"After large area removal: {len(np.unique(masks))-1}")
    stitched_masks, df_with_centroids = post.bipartite_stitching(masks, verbose=verbose)
    if verbose >= 1:
        print(f"After stitching: {len(np.unique(stitched_masks))-1}")
    neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                verbose-1)
    if verbose >= 1:
        print(f"After splitting: {len(np.unique(split_masks))-1}")

    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        post.remove_short_neurons(split_masks,
                                  split_lengths,
                                  lower_length_threshold,
                                  split_brightnesses,
                                  split_neuron_planes)
    if verbose >= 1:
        print(f"After short neuron removal: {len(np.unique(final_masks))-1}")

    if to_remove_border is True:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    if verbose >= 1:
        print(f"After border removal: {len(np.unique(final_masks))}")
        print("Postprocessing finished")

    return final_masks


def perform_post_processing_3d(stitched_masks, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3, verbose=0):
    """
    Performs post-processing of segmented masks. Includes: splitting long masks, removing large areas,
    removing short masks as well as removing artefacts close to the border

    Parameters
    ----------
    stitched_masks
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    border_width_to_remove : int
        within that distance to border, artefacts/masks will be removed
    to_remove_border : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    labels : 3D numpy array
        3D array of processed masks
    """

    stitched_masks = post.remove_large_areas(stitched_masks)
    neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths, verbose=verbose-1)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                verbose=verbose-1)

    # remove short neurons
    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        post.remove_short_neurons(split_masks,
                                  split_lengths,
                                  lower_length_threshold,
                                  split_brightnesses,
                                  split_neuron_planes)

    if to_remove_border is True:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    return final_masks
