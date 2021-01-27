import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import os

def calc_all_overlaps(all_2d_masks: list,
                      verbose=0):
    """
    Get the "tube" of a neuron through z slices via most overlapping pixels

    Parameters
    ----------
    all_2d_masks : list
        List of all masks across z

    See also: calc_best_overlap
    """
    num_slices = len(all_2d_masks)

    # Final output matrix: one big mask
    # Syntax: ZXY
    sz = (num_slices, ) + all_2d_masks[0].shape
    full_3d_mask = np.zeros(sz)

    # Initial neuron and mask
    global_current_neuron = 1
    hist_dict = {}

    # Loop over slices to start (2d masks)
    for i_slice in range(num_slices-1):
        this_mask_all_neurons = all_2d_masks[i_slice]
        all_neurons_this_mask = np.unique(this_mask_all_neurons)

        if verbose >= 1:
            print(f"Found {len(all_neurons_this_mask)} neurons on slice {i_slice}")

        # Pick a neuron, and finalize it by looping over ALL later slices
        for this_neuron in all_neurons_this_mask:
            # skip background value
            if this_neuron == 0:
                continue

            # Get the initial mask, which will be propagated across slices
            this_mask_binary = (this_mask_all_neurons==this_neuron)

            for i_next_slice in range(i_slice+1, num_slices):
                next_mask_all_neurons = all_2d_masks[i_next_slice]
                len_of_current_neuron = i_next_slice - i_slice + 1

            # Get best overlap between the current mask and the next slice
                this_overlap, next_mask_binary = calc_best_overlap(this_mask_binary, next_mask_all_neurons)

                # If good enough, save in the master 3d mask
                min_overlap = calc_min_overlap()
                if not min_overlap:
                    min_overlap = 10

                if this_overlap > min_overlap:
                    full_3d_mask[i_next_slice, : ,:][next_mask_binary] = global_current_neuron
                    # zero out used neurons on slice
                    all_2d_masks[i_next_slice][next_mask_binary] = 0
                    this_mask_binary = next_mask_binary

                else:
                    len_of_current_neuron = i_next_slice - i_slice + 1
                    # TODO for hist_dict: find a way to figure out, whether there has been any match found on last slice. Otherwise len-1
                    hist_dict[str(global_current_neuron)] = len_of_current_neuron - 1

                    # If no overlap, start the next neuron
                    global_current_neuron += 1
                    if verbose >= 1:
                        print(f"Finished neuron {global_current_neuron}")
                        print(f"Includes {len_of_current_neuron} slices")
                    break

        # get all neuron IDs and best overlaps of a given neuron mask across all other masks
        # all_neurons[i], all_overlaps[i], this_mask = calc_best_overlap(prev_mask, this_mask)
        # min_overlap = calc_min_overlap()
        # if all_overlaps[i] <= min_overlap:
        #     # Start a new neuron
        #     break
        # all_masks.append(this_mask)

    # TODO histogram of neuron length in Z (<4 = incorrect)

    # TODO

    # maybe save file as TIFF
    # TODO decide on file format (tiff or numpy)
    return full_3d_mask



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
        Masks of all neurons detected in the next slice

    """
    best_overlap = 0
    best_ind = None
    best_mask = np.zeros_like(masks_s1)
    all_vals = np.unique(masks_s1)
    for i, val in enumerate(all_vals):
        if val == 0:
            continue
        this_neuron_mask = masks_s1==val
        overlap = np.count_nonzero(mask_s0*this_neuron_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ind = i
            best_mask = this_neuron_mask
    if verbose >= 2:
        print(f'Best Neuron: {best_ind}, overlap of {best_overlap} between it and original')

    return best_overlap, best_mask


def calc_min_overlap():
    """
    TODO: make this a function of track length? or of neuron size?
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

    all_2d_masks = []

    for i, file in enumerate(files_list):
        slice = np.load(file)
        # create a list of 2d masks to pass tpo calc_all_overlap
        all_2d_masks.append(slice)

    full_3d_mask = calc_all_overlaps(all_2d_masks, 2)

    return masks_3d

# print(f'starting the overlapping')
# example_input = r'C:\Users\niklas.khoss\Desktop\stardist_testdata'
# stitched_3d_output = convert_to_3d(example_input)
# print(f'3d output shape: {stitched_3d_output.shape}')