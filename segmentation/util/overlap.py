"""
Finds 2d masks in a given path and returns a 3d array with consecutive unique IDs/values for every
matched neuron, i.e. if a neuron (mask) was overlapping in neighbouring slices, the masks will get the same value.
For now, it only supports '.npy' arrays.

Usage:
    convert_to_3d(path_to_2d_masks)

Output:
    3d-numpy array

"""
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import tifffile as tiff
import os
from collections import defaultdict
from tqdm import tqdm

def calc_all_overlaps(array_3d,
                      verbose=0):
    """
    Get the "tube" of a neuron through z slices via most overlapping pixels

    Parameters
    ----------
    array_3d : numpy array
         Numpy 3d array of a volume
    verbose:    int
        Flag for printing more information.

    Returns
    -------
    3d array with masks of neurons. Masks were connected in 3d and given unique new IDs (i.e. value in array)
    Dictionary of neuron lengths per neurons
    Dictionary of average brightness per neuron mask and slice. (i.e. 'Neuron1': [100(z=0), 120(z=1),...])

    See also: calc_best_overlap
    """
    # TODO: Debug the following problem:
    #  when using Stardist data, the stitched array does not contain a neuron 9, although the neuron length dict has one

    print(f'Starting with stitching. Array shape: {array_3d.shape}')
    num_slices = len(array_3d)

    # Initialize output matrix: full_3d_mask
    # Dimensions: ZXY
    full_3d_mask = np.zeros_like(array_3d)

    # Initial neuron and mask
    global_current_neuron = 1
    hist_dict = {}
    brightness_dict = defaultdict(list)

    # Loop over slices to start (2d masks)
    for i_slice in range(num_slices-1):
        this_mask_all_neurons = array_3d[i_slice]
        all_neurons_this_mask = np.unique(this_mask_all_neurons)

        if verbose >= 1:
            print(f"Found {len(all_neurons_this_mask)} neurons on slice {i_slice}")

        # Pick a neuron, and finalize it by looping over ALL later slices
        for this_neuron in all_neurons_this_mask:
            # skip background value
            if this_neuron == 0:
                continue

            # Get the initial mask, which will be propagated across slices
            this_mask_binary = (this_mask_all_neurons == this_neuron)

            # create correct masks for slice 0
            if i_slice == 0:
                first_slice = full_3d_mask[0]
                first_slice[this_mask_binary] = global_current_neuron
                full_3d_mask[0] = first_slice

            for i_next_slice in range(i_slice+1, num_slices):
                next_mask_all_neurons = array_3d[i_next_slice]
                len_of_current_neuron = i_next_slice - i_slice + 1

                # Get best overlap between the current mask and the next slice
                this_overlap, next_mask_binary = calc_best_overlap(this_mask_binary, next_mask_all_neurons)

                # If good enough, save in the master 3d mask
                min_overlap = calc_min_overlap()

                is_good_enough = (this_overlap > min_overlap)
                if is_good_enough:
                    tmp = full_3d_mask[i_next_slice, ...]
                    tmp[next_mask_binary] = global_current_neuron
                    full_3d_mask[i_next_slice, ...] = tmp

                    # zero out used neurons on slice
                    array_3d[i_next_slice][next_mask_binary] = 0
                    this_mask_binary = next_mask_binary


                # Finalize this neuron, and move to next neuron
                is_on_last_slice = i_next_slice==(num_slices-1)
                if not is_good_enough or is_on_last_slice:
                    len_of_current_neuron = i_next_slice - i_slice + 1

                    if len_of_current_neuron <= 2:
                        # add the mask of current neuron to current slice, if it was short (l=1)
                        interim_slice = full_3d_mask[i_slice]
                        interim_slice[this_mask_binary] = global_current_neuron
                        full_3d_mask[i_slice] = interim_slice

                    if is_on_last_slice:
                        hist_dict[str(global_current_neuron)] = len_of_current_neuron
                    else:
                        hist_dict[str(global_current_neuron)] = len_of_current_neuron - 1

                    global_current_neuron += 1
                    if verbose >= 1:
                        print(f"Finished neuron {global_current_neuron}")
                        print(f"Includes {len_of_current_neuron} slices")
                    break

    # maybe save file as TIFF
    # TODO decide on file format (tiff or numpy)

    # TODO visualize 3d in fiji

    print(f'end of calc_all_overlaps: full_3d non-zeros: {np.count_nonzero(full_3d_mask)}')
    print(f'Shape: {full_3d_mask.shape}')
    return full_3d_mask, hist_dict


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
    default_min = 1
    return default_min


def convert_to_3d(files_path: str, verbose=0):
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
    print(f'Start of overlap')

    # check, if input str is a valid directory
    if not os.path.isdir(files_path):
        print(f'convert to 3d: {files_path} is no directory!')
        return False

    # check, if folder contains npy files and if, find all npy files
    files_list = [os.path.join(files_path, x.name) for x in os.scandir(files_path) if x.name.endswith('.npy')]
    # TODO add some checks for folder content

    if not files_list:
        print(f'There are no .npy files in {files_path}')
        return False
    files_list = natsorted(files_list)

    # iterate over files, load the mask and concatenate them into a 3D array
    # initialize output array (ZXY): size = (#files, file.shape)
    slice_for_size = np.load(files_list[0])
    size_3d = (len(files_list), ) + (slice_for_size.shape)      # tuple addition

    all_2d_masks = []

    for i, file in enumerate(files_list):
        slice = np.load(file)
        # create a list of 2d masks to pass to calc_all_overlap
        all_2d_masks.append(slice)

    stitched_3d_masks, neuron_lengths = calc_all_overlaps(all_2d_masks, verbose)
    # TODO post-processing: brightness, long-neuron-split, short neuron removal
    # make a new loop to process the above

    # TODO: top level function (path to data) -> create 3d array if not already there THEN next-level function (3d array) -> processing

    f1, f2 = neuron_length_hist(neuron_lengths)

    return stitched_3d_masks, neuron_lengths


# histogram of neuron 'lengths' across Z
def neuron_length_hist(lengths_dict: dict, save_flag=0):
    # plots the lengths of neurons in a histogram and barplot
    vals = lengths_dict.values()
    keys = lengths_dict.keys()

    fig = plt.figure()
    plt.hist(vals, bins=np.arange(1, max(vals)+2), align='left')
    plt.xticks(np.arange(1, max(vals) + 2))
    # TODO add automated y-axis limits for histograms
    #plt.yticks(np.arange(0, 27, 2))
    plt.xlabel('neuron length')
    plt.ylabel('# of neurons')
    plt.title('neuron lengths histogram')
    plt.show()

    if save_flag >= 1:
        plt.savefig(r'C:\Segmentation_working_area\neuron_lengths\example_results_n_lengths.png')

    fig1 = plt.figure()
    plt.bar(keys, vals)
    plt.title('Neuron lengths per neuron')
    plt.ylabel('Length')
    plt.xlabel('Neuron #')

    if save_flag >= 1:
        plt.savefig(r'C:\Segmentation_working_area\neuron_lengths\example_results_n_lengths_bar.png')

    return fig, fig1

def remove_short_neurons(array, neuron_lengths, length_cutoff):
    # remove all neurons, which are too short (e.g. < 3)

    rm_list = list()
    for key, value in neuron_lengths.items():
        if int(value) < length_cutoff:
            print(f'removed {key}')
            # remove from 3d array
            cull = array == int(key)
            array[cull] = 0

            rm_list.append(key)

    # remove entry from dictionary
    for r in rm_list:
        del neuron_lengths[str(r)]

    return array, neuron_lengths, rm_list


def split_long_neurons(array, neuron_lengths: dict, neuron_brightnesses, global_current_neuron):
    # if a neuron is too long (>12 slices), it will be cut off and a new neuron will be initialized

    neuron_brightnesses = calc_brightness(array)

    # iterate over new_indices dict and split neurons accordingly. Change current neuron length and
    # append new neurons at end of dict
    # for key, value in neuron_lengths.items():
    #     if len_counter > max_neuron_length:
    #         # add the neuron to the lenghts-dict
    #         neuron_lengths[str(global_current_neuron)] = len_counter - 1 #len_of_current_neuron
    #         print(f'Neuron {global_current_neuron} > 12')
    #         global_current_neuron += 1
    #         len_counter = 1

            # use split_one_neuron in loop, if logic says too long
    return array, neuron_lengths
    pass

def split_one_neuron(brightness_across_z):
    # decides, whether a neuron is too long according to brightness distribution
    # return new_indices
    return

def calc_brightness(original_array, stitched_masks, neuron_lengths):
    # TODO: add brightness (avg per mask per slice) to a dict {'global neuron': [list of brightnesses]}
    print('Start with brightness calculations')
    # add default dict
    brightness_dict = defaultdict(list)

    # loop over the actual data and calculate average brightness per neuron per slice/mask
    for neuron in neuron_lengths.keys():
        current_list = list()

        for i_slice, slice in enumerate(stitched_masks):
            # get the mask
            if int(neuron) in slice:
                this_mask = slice == int(neuron)

                # get the average brightness for that mask
                current_brightness = int(np.nanmean(original_array[i_slice, this_mask]))

                # extend the brightness dict
                if current_brightness is not np.nan:
                    current_list.append(current_brightness)
                else:
                    print(f'NaN in neuron {neuron} slice {i_slice}')

                # TODO can be optimized by breaking the loop after not finding anything
                #  (but beginning slice needs to be figured out)

        # add list of brightnesses to dict
        brightness_dict[neuron].extend(current_list)

    print(f'Brightness: {len(brightness_dict)}    Masks: {len(neuron_lengths)}')
    print(f'Done with brightness')
    return brightness_dict


def create_3d_array(files_path, verbose=0):
    """
    Creates a 3D numpy array from many 2D arrays in a folder. It concatenates them, so that dimensions are ZXY.
    Assumes only connected npy arrays in same folder and 1 volume per folder (for now), but it will just concatenate
    all files in a natural sorted manner according to their filenames.

    Parameters
    ----------
    files_path: str
        String of file path from root
    verbose:    int
        Flag for more information printing
    Returns
    -------
    Returns a 3D numpy array with dimensions ZXY.

    """
    if not os.path.isdir(files_path):
        print(f'convert to 3d: {files_path} is no directory!')
        return False

    # check, if folder contains npy files and if, find all npy files
    files_list = [os.path.join(files_path, x.name) for x in os.scandir(files_path) if x.name.endswith('.npy')]

    if not files_list:
        print(f'There are no .npy files in {files_path}')
        return False
    files_list = natsorted(files_list)

    # if 1 file => 3D array => return; else create 3D array
    if len(files_list) == 1:
        arr = np.load(files_list[0])
        print(f'Found 1 file in {files_path} with shape: {arr.shape}')

        if arr.shape[0] < 2:
            print(f'... BUT there are < 2 Z planes in {files_list[0]}. Will return False')
            return False
        else:
            return arr
    else:
        if verbose >= 1:
            print(f'Found {len(files_list)} files in {files_path}')

        # iterate over files, load the mask and concatenate them into a 3D array
        # initialize output array (ZXY): size = (#files, file.shape)
        slice_for_size = np.load(files_list[0])
        size_3d = (len(files_list), ) + (slice_for_size.shape)      # tuple addition

        # initialize 3d array; need that slice as seed for stacking
        array_3d = np.zeros(size_3d)

        for i, file in enumerate(files_list):
            slice = np.load(file)
            array_3d[i, ...] = slice

        if verbose >= 1:
            print(f'Shape of output array: {array_3d.shape}')
        return array_3d

def create_3d_array_from_tiff(img_path: str, flyback_flag=1):
    """
    Creates a 3D array from a 3D tiff file. Made for one volume (!), but it will concatenate all tif-files within
    the folder in a natural sorted manner according to filenames.
    Dimensions will be ZXY.

    Parameters
    ----------
    img_path:   str
        String of path to folder containing tif file
    flyback_flag: int
        0 or 1; determines the first slice to be added to the 3D array. Default is 1 -> omitting z=0!

    Returns
    -------
    3D numpy array with dimensions ZXY

    """
    print(f'Creating 3D npy array from imaging data: {img_path}')

    if not os.path.isdir(img_path):
        print(f'convert to 3d: {img_path} is no directory!')
        return False

    # check, if folder contains npy files and if, find all npy files
    img_list = [os.path.join(img_path, x.name) for x in os.scandir(img_path) if x.name.endswith('.tif')]

    if not img_list:
        print(f'There are no .tif files in {img_path}')
        return False

    with tiff.TiffFile(img_list[0]) as my_tiff:
        size = (len(my_tiff.pages), *my_tiff.pages[0].asarray().shape)  # '*' to unpack tuple right away
        img_3d_array = np.zeros(size)

        for p, page in enumerate(my_tiff.pages):
            img_3d_array[p] = page.asarray()

    # skipping first plane if flyback == 1
    if flyback_flag:
        img_3d_array = img_3d_array[1:, ...]

    return img_3d_array

# TODO: write a new main function, which can discern between 2d & 3d data and call overlaps
def main_overlap(img_data_path, algo_data_path):

    # F1: load/create 3d array of algorithm results and original imaging data
    algo_array_3d = create_3d_array(algo_data_path)

    # TODO: need to change the call for the imaging data! We should not have to load and convert it for every volume!
    img_array_3d = create_3d_array_from_tiff(img_data_path)   # original imaging data as 3d array

    # F2: overlap + post-processing
    end_results = level2_overlap(img_array_3d, algo_array_3d)

    pass

def level2_overlap(img_array, algo_array):
    # this function shall call the necessary calculation functions etc
    # returns the mask_arrays, neuron lengths, brightnesses, etc

    stitched_3d_masks, neuron_lengths = calc_all_overlaps(algo_array)

    # calculate average brightness per neuron mask
    brightness_dict = calc_brightness(img_array, stitched_3d_masks, neuron_lengths)

    # split long neurons

    # remove short neurons

    # neuron length histogram
    h1, h2 = neuron_length_hist(neuron_lengths)


    return

# gt_path = r'C:\Segmentation_working_area\gt_masks_npy'
# sd_path = r'C:\Segmentation_working_area\stardist_testdata\masks'
# img_data_path = r'C:\Segmentation_working_area\test_volume'
#
# # sd_3d_stitched = np.load(r'C:\Segmentation_working_area\stitched_3d_data\stardist_fluo_stitched_3d.npy')
#
# og_3d = create_3d_array_from_tiff(img_data_path)
#
# sd_3d = create_3d_array(sd_path)
# sd_stitch, sd_nlen = calc_all_overlaps(sd_3d)
# sd_bright = calc_brightness(og_3d, sd_stitch, sd_nlen)
# print('end')
