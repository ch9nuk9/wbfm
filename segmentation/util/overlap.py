import numpy as np
from natsort import natsorted
import os

<<<<<<< HEAD
def calc_all_overlaps(all_2d_masks,
=======
# 'stitching'
# top level function should get pat h as input and call 'calc_all_overlaps'
# in calc_all_overlaps, call a helper function to get list of
def calc_all_overlaps(start_neuron,
                      all_2d_masks,
>>>>>>> 64c5f8521c5fee3401123f8863d12d0584590e88
                      verbose=1):
    """
    Get the "tube" of a neuron through z slices via most overlapping pixels

    Parameters
    ----------
    all_2d_masks : list
        List of all masks across z

    See also: calc_best_overlap
    """
    num_slices = len(all_2d_masks)

    # Syntax: ZXY
    sz = (num_slices, ) + all_2d_masks[0].shape
    full_3d_mask = np.zeros(sz)

    # Initial neuron and mask
	i_neuron = 1
    all_masks.append(all_2d_masks[0] == start_neuron)
    all_neurons[0] = start_neuron

    for i, this_mask in enumerate(all_2d_masks):
        if i==0:
            continue
        # Retain the last successful mask
        prev_mask = all_2d_masks[i-1] == all_neurons[i-1]

        # get all neuron IDs and best overlaps of a given neuron mask across all other masks
        all_neurons[i], all_overlaps[i], this_mask = calc_best_overlap(prev_mask, this_mask)
        min_overlap = calc_min_overlap()
        if all_overlaps[i] <= min_overlap:
            # Start a new neuron
            break
        all_masks.append(this_mask)

    return all_neurons, all_overlaps, all_masks



def calc_best_overlap(mask_s0, # Only one mask
                      masks_s1,
                      verbose=1):
    """
    Calculates the best overlap between an initial mask and all subsequent masks
        Note: calculates pairwise for adjacent time points

    Parameters
    ----------
    mask_s0 : array_like
        Mask of the original neuron
    masks_s1 : list
        Masks of all neurons detected in the next frame

    """
    best_overlap = 0
    best_ind = None
    best_mask = np.zeros_like(masks_s1)
    all_vals = np.unique(masks_s1)
    for i,val in enumerate(all_vals):
        if val == 0:
            continue
        this_neuron_mask = masks_s1==val
        overlap = np.count_nonzero(mask_s0*this_neuron_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ind = i
            best_mask = this_neuron_mask
    if verbose >= 2:
        print(f'Best Neuron: {best_ind}, overlap between {best_overlap} and original')
    return best_ind, best_overlap, best_mask


def calc_min_overlap():
    """
    TODO: make this a function of track length?
    """
    return 0.0


def convert_to_3d(files_path: str):
    """
    Converts all 2d numpy arrays within a folder to a 3d array. Throws error if not a dir or no npy files within.

    Parameters
    ----------
    files_path : str
        path of directory containing .npy arrays of masks

    Returns
    -------
    masks_3d : 3d numpy array
        a 3d array of the concatenated masks from a segmentation algorithm (e.g. stardist)

    """
    # check, if input str is a valid directory
    if not os.path.isdir(files_path):
        print(f'convert to 3d: {files_path} is no directory!')
        return False

    # check, if folder contains npy files and if, find all npy files
    files_list = [os.path.join(files_path, x.name) for x in os.scandir(files_path) if x.name.endswith('.npy')]

    if not files_list:
        print(f'There are no .npy files in {files_path}')
        return False
    files_list = natsorted(files_list)

    # iterate over files, load the mask and concatenate them into a 3D array
    # initialize output array (ZXY): size = (#files, file.shape)
    slice_for_size = np.load(files_list[0])
    size_3d = (len(files_list), ) + (slice_for_size.shape)      # tuple addition
    masks_3d = np.zeros(size_3d, dtype=np.int8)

    # print(f'masks_3d size: {masks_3d.shape}')

    for i, file in enumerate(files_list):
        slice = np.load(file)
        masks_3d[i] = slice     # appending this way instead of np.dstack or concatenate (need a 3d-array as seed)

    return masks_3d
