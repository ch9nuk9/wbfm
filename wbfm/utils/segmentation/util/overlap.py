"""
Finds 2d masks in a given path and returns a 3d array with consecutive unique IDs/values for every
matched neuron, i.e. if a neuron (mask) was overlapping in neighbouring slices, the masks will get the same value.
For now, it only supports '.npy' arrays.

Usage:
    convert_to_3d(path_to_2d_masks)

Output:
    3d-numpy array

"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from wbfm.utils.general.utils_networkx import calc_bipartite_matches_using_networkx
from wbfm.utils.tracklets.utils_tracklets import build_tracklets_from_matches
from natsort import natsorted
from scipy.optimize import curve_fit

import wbfm.utils.segmentation.util.utils_model as sd


def calc_all_overlaps(array_3d,
                      verbose=0):
    """
    Outdated! Use 'bipartite_stitching' instead!

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
    # TODO: improve speed by using views of slices (= sub-areas within a slice) instead of whole slices

    print(f'Starting with stitching. Array shape: {array_3d.shape}')
    num_slices = len(array_3d)

    # Initialize output matrix: full_3d_mask
    # Dimensions: ZXY
    full_3d_mask = np.zeros_like(array_3d)

    # Initial neuron and mask
    global_current_neuron = 1
    hist_dict = {}

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
                        hist_dict[global_current_neuron] = len_of_current_neuron
                    else:
                        hist_dict[global_current_neuron] = len_of_current_neuron - 1

                    global_current_neuron += 1
                    if verbose >= 1:
                        print(f"Finished neuron {global_current_neuron}")
                        print(f"Includes {len_of_current_neuron} slices")
                    break

    # maybe save file as TIFF
    # TODO decide on file format (tiff or numpy)

    # TODO visualize 3d in fiji

    # TODO think about adding the start-plane to neuron length dict!

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


def bipartite_stitching(array_3d, num_slices=0, verbose=0):
    """
    The function tries to connect segmented masks of adjacent planes, which overlap when planes are projected
    onto each other. The goal is a 3D array with a unique ID for every neuron.
    Iterates over slices and their neurons and calculates the best matches using
    the bipartite matching algorithm of graph theory.

    Parameters
    ----------
    array_3d : 3D numpy array
        3D array of segmented masks with recurring (i.e. non-unique) neuron IDs across planes
    num_slices : int
        Number of slices per volume (here: array_3d)
    verbose : int
        Flag for printing extra information

    Returns
    -------
    sorted_stitched_array : 3D numpy array
        3D array with connected and unique neuron IDs across planes
    clust_df : pandas dataframe
        Contains extra information about each neuron and their prospective matches
    all_centroids : list of lists
        contains all centroids found across al planes
    all_matches : list of lists
        contains all tentative matches found by the bipartite matching algorithm
    """

    print(f'Starting with stitching. Array shape: {array_3d.shape}')
    # TODO get num_slices from an argument!
    num_slices = len(array_3d)

    # Initialize output matrix
    all_matches = []
    all_centroids = []

    # loop over slices
    for i_slice in range(num_slices):

        this_slice = array_3d[i_slice]
        bp_matches = []

        if i_slice < num_slices - 1:
            next_slice = array_3d[i_slice + 1]
            this_slice_candidates = list()
            this_slice_candidates = create_matches_list(this_slice, next_slice)

            # Bipartite matching after creating overlap list for all neurons on slice
            bp_matches = list()
            bp_matches = sorted(calc_bipartite_matches_using_networkx(this_slice_candidates))

            all_matches.append(bp_matches)

        # get centroid coordinates for all found neurons/masks
        these_centroids = []
        for this_neuron in range(int(np.amax(this_slice)) + 1):
            this_x, this_y = np.where(this_slice == this_neuron)

            if len(this_x) == 0:
                # negative location values for unusable neurons
                these_centroids.append([-15, -15, -15])
            else:
                these_centroids.append([i_slice, round(np.mean(this_x)), round(np.mean(this_y))])

        all_centroids.append(these_centroids)

    clust_df = build_tracklets_from_matches(all_centroids, all_matches)

    # renaming all found neurons in array; in a sorted manner
    sorted_stitched_array = renaming_stitched_array(array_3d, clust_df)

    return sorted_stitched_array, (clust_df, all_centroids, all_matches)


def create_matches_list(slice_1, slice_2, verbose=0):
    """
    Creates a list of lists with all matches between slice_1 and slice_2. A match is counted, when a neuron mask on
    slice 1 overlaps with another neuron mask on slice_2 by at least 1 pixel!

    Parameters
    ----------
    slice_1 : 2D numpy array
    slice_2 : 2D numpy array
    verbose : int
        Flag for printing extra information

    Returns
    -------
    bip_list : list of lists
        list containing all matches on slice_2 for all neurons on slice_1 and the area, by which they overlap
    """

    # find all matches of a given neuron in the next slice
    neurons_this_slice = np.unique(slice_1)
    bip_list = list()

    # iterate over all neurons found in array[i_slice]
    for this_neuron in neurons_this_slice:
        bip_inter = list()
        if this_neuron == 0:
            continue

        if verbose >= 2:
            print(f'... Neuron: {int(this_neuron)}')

        # Get the initial mask, which will be propagated across slices
        this_mask_binary = (slice_1 == this_neuron)

        # overlap
        this_overlap_neurons = np.unique(slice_2[this_mask_binary])

        for overlap_neuron in this_overlap_neurons:
            if overlap_neuron == 0:
                continue
            overlap_slice = slice_2 == overlap_neuron
            overlap_slice = this_mask_binary[overlap_slice]
            overlap_area = np.count_nonzero(overlap_slice)

            bip_inter.append([int(this_neuron), int(overlap_neuron), int(overlap_area)])

        bip_list.extend(bip_inter)

    return bip_list


def renaming_stitched_array(arr, df):
    """
    Takes an array and changes the values of masks, so that it starts at 1 on slice 1 and
    increases consistently/consecutively across planes with an unique value for each mask.
    It uses Charlie's generated dataframe (from build_tracklets_from_matches()) to rename the masks.

    Parameters
    ----------
    arr : numpy array (3d)
        3D array of masks with recurring valuesacross planes
    df : pandas dataframe
        dataframe containing extra information for each mask match across Z

    Returns
    -------
    sorted array : 3D numpy array
        3D array of masks with unique values across Z
    """

    print(f'Starting to rename stitched array using Charlies dataframe')
    renamed_array = np.zeros_like(arr)

    # now, change ALL local indices on slice_ind to clust_ind + 1 to create a new neuron
    for i, row in df.iterrows():
        for elem, og_slice in enumerate(row['slice_ind']):
            renamed_array[og_slice, arr[og_slice] == row['all_ind_local'][elem]] = row['clust_ind'] + 1

    return renamed_array


def neuron_length_hist(lengths_dict: dict, save_path='', save_flag=0):
    # plots the lengths of neurons in a histogram and barplot
    vals = lengths_dict.values()
    keys = lengths_dict.keys()
    if save_path:
        fname = os.path.split(save_path)[1]
    else:
        fname=''

    fig = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.hist(vals, bins=np.arange(1, max(vals)+2), align='left')
    plt.xticks(np.arange(1, max(vals) + 2))
    # TODO add automated y-axis limits for histograms
    #plt.yticks(np.arange(0, 27, 2))
    plt.xlabel('neuron length')
    plt.ylabel('# of neurons')
    plt.title('neuron lengths histogram ' + fname)
    # plt.show()

    # TODO: change save folder
    if save_flag >= 1:
        plt.savefig(os.path.join(save_path, fname + '_neuron_lengths.png'), dpi=96)

    fig1 = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.bar(keys, vals)
    plt.title('Neuron lengths per neuron ' + fname)
    plt.ylabel('Length')
    plt.xlabel('Neuron #')

    if save_flag >= 1:
        plt.savefig(os.path.join(save_path, fname + '_neuron_lengths_bar.png'), dpi=96)

    return fig, fig1


def remove_short_neurons(array, neuron_lengths, length_cutoff, brightness, neuron_planes):
    """
    Removes neurons from array, if they are shorter than 'length_cutoff'.
    Also removes the entries of these neurons from all dictionaries (length, brightness, global_z.

    Parameters
    ----------
    array : 3D numpy array
        3D array with segmented masks; neurons have unique IDs
    neuron_lengths : dict(int)
        Contains the length of each neuron
        neuron_ID = 1
        neuron_lengths[neuron_ID] == [3]
    length_cutoff : int
        Shortest a neuron may be
    brightness : dict(list)
        Contains the average brightness values per plane of each neuron
        neuron_ID = 1
        brightness[neuron_ID] == [250, 340, 220]
    neuron_planes : dict(list)
        Contains the Z-planes, on which a neuron was found
        neuron_ID = 1
        neuron_z_planes[neuron_ID] == [12, 13, 14]

    Returns
    -------
    array : 3D numpy array
        array with short neurons removed
    neuron_lengths : dict(int)
        dict entries of short neurons removed
    brightness : dict(list)
        dict entries of short neurons removed
    neuron_planes : dict(list)
        dict entries of short neurons removed
    rm_list : list
        list with indices of removed neurons
    """

    # remove all neurons, which are too short (e.g. < 3)
    # TODO: remove short neurons from brightness dict too!
    rm_list = list()
    for key, value in neuron_lengths.items():
        if value < length_cutoff:
            # print(f'removed {key}')
            # remove from 3d array
            cull = array == key
            array[cull] = 0

            rm_list.append(key)

    # remove entry from dictionary
    for r in rm_list:
        del neuron_lengths[r]
        del brightness[r]
        del neuron_planes[r]

    return array, neuron_lengths, brightness, neuron_planes, rm_list


def split_long_neurons(array,
                       neuron_lengths: dict,
                       neuron_brightnesses: dict,
                       global_current_neuron,
                       maximum_length,
                       neuron_z_planes: dict):
    """
    Splits neuron, which are too long (to our understanding) into 2 parts. The split-point is the midpoint between
    2 gaussians, that have been fit onto a plot of the average brightness (per slice) of that neuron. The second half
    of the split neuron gets a new ID and will be appended.

    Parameters
    ----------
    array : 3D numpy array
        Array of segmented masks with unique IDs
    neuron_lengths : dict(list)
        Contains the lengths of each neuron found in array
        neuron_ID = 1
        neuron_lengths[neuron_ID] == [3]
    neuron_brightnesses : dict(list)
        Contains the average brightness per plane of each neuron
        neuron_ID = 1
        neuron_brightnesses[neuron_ID] == [250, 340, 225]
    global_current_neuron : int
        Highest neuron ID in dataset + 1
    maximum_length : int
        Threshold for neuron length
    neuron_z_planes : dict(list)
        Contains the Z-planes corresponding to the brightness values of each neuron
        neuron_ID = 1
        neuron_z_planes[neuron_ID] == [12, 13, 14]

    For Tests:
    ----------
    -   the dictionary values (lengths, brightnesses & brightnesses_z) may not be longer than the values of the
        input dictionary was!
    -   all three dictionaries must be of same length

    Returns
    -------
    array : 3D numpy array
        3D array with split neurons
    neuron_lengths : dict(list)
        dict containing neuron lengths and new entries for split neurons
    neuron_brightnesses : dict(list)
        dict containing average neuron brightnesses per plane with new entries for split neurons
    global_current_neuron :
        new highest ID/number for neurons
    neuron_z_planes : dict(list)
        dict with list of z-planes, in which a neuron was found; new entries for split neurons

    """
    # if a neuron is too long (>12 slices), it will be cut off and a new neuron will be initialized

    # iterate over neuron lengths dict, and if z >= 12, try to split it
    # TODO iterate over the dictionary itself
    for i in range(1, len(neuron_lengths) + 1):
        if neuron_lengths[i] > maximum_length:

            try:
                x_means, _, _ = calc_means_via_brightnesses(neuron_brightnesses[i])
            except ValueError:
                print(f'! ValueError while splitting neuron {i}. Probably could not fit 2 Gaussians! Will continue.')
                continue
            # if neuron can be split
            if x_means:
                x_split = round(sum(x_means) / 2)
                # print(f'Splitting neuron {i} at {x_split}, new neuron {global_current_neuron + 1}')

                # create new entry
                global_current_neuron += 1
                neuron_lengths[global_current_neuron] = neuron_lengths[i] - x_split - 1

                # update neuron lengths and brightnesses entries; 0-x_split = neuron 1
                neuron_lengths[i] = x_split + 1

                # update mask array with new mask IDs
                for i_plane, plane in enumerate(array[x_split:]):
                    if i in plane:
                        inter_plane = plane == i
                        plane[inter_plane] = global_current_neuron

                # update brightnesses and brightness-planes dicts
                neuron_brightnesses[global_current_neuron] = neuron_brightnesses[i][x_split + 1:]
                neuron_brightnesses[i] = neuron_brightnesses[i][:x_split]

                neuron_z_planes[global_current_neuron] = neuron_z_planes[i][x_split + 1:]
                neuron_z_planes[i] = neuron_z_planes[i][:x_split]

            else:
                print(f'Could not split neuron {i}, although it is longer than {maximum_length}')

    return array, neuron_lengths, neuron_brightnesses, global_current_neuron, neuron_z_planes


def calc_brightness(original_array, stitched_masks, neuron_lengths):
    """
    Calculates the average brightness of each mask per plane in an array given the original image
    and the mask lengths.

    Parameters
    ----------
    original_array : 3D numpy array
        original image array
    stitched_masks : 3D numpy array
        array with stitched (=unique and consecutive values per neuron) neuron masks
    neuron_lengths : dict
        dictionary with neuron lengths in Z for each neuron
        {neuron_ID: length (int)}
        neuron_lengths[1] == [12, 13, 14]
        neuron_ID = 1
        neuron_lengths[neuron_ID] # [12, 13, 14]

    Returns
    -------
    brightness_dict : dict
        contains a list of average brightness values per plane for each neuron
        {neuron #: [average brightnesses per plane]}
    brightness_planes : dict
        contains the global Z plane index for each neuron
        {neuron #: [global Z planes]}

    """

    print('Start with brightness calculations')
    # add default dict
    brightness_dict = defaultdict(list)
    brightness_planes = defaultdict(list)

    # loop over the actual data and calculate average brightness per neuron per slice/mask
    for neuron in neuron_lengths.keys():
        current_list = list()
        planes_list = list()

        for i_slice, slice in enumerate(stitched_masks):
            # get the mask
            if int(neuron) in slice:
                this_mask = slice == int(neuron)

                # get the average brightness for that mask
                current_brightness = int(np.nanmean(original_array[i_slice, this_mask]))

                # extend the brightness dict
                if current_brightness is not np.nan:
                    current_list.append(current_brightness)
                    planes_list.append(i_slice)
                else:
                    print(f'NaN in neuron {neuron} slice {i_slice}')

                # TODO can be optimized by breaking the loop after not finding anything
                #  (but beginning slice needs to be figured out)

        # add list of brightnesses to dict
        brightness_dict[neuron].extend(current_list)
        brightness_planes[neuron].extend(planes_list)

    print(f'Brightness: {len(brightness_dict)}    Masks: {len(neuron_lengths)}')
    print(f'Done with brightness')
    return brightness_dict, brightness_planes


def calc_means_via_brightnesses(brightnesses, plots=0):
    # calculate the means of 2 underlying neuron brightness distributions

    y_data = np.array(brightnesses)
    x_data = np.array(np.arange(len(y_data)))

    # Define model function to be used to fit to the data above:
    # Adapt it to as many gaussians you may want
    # by copying the function with different A2,mu2,sigma2 parameters

    def gauss2(x, *p):
        A1, mu1, sigma1, A2, mu2, sigma2 = p
        return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

    # p0 is the initial guess for the fitting coefficients
    # initialize them differently so the optimization algorithm works better
    height = len(y_data)/4
    p0 = [np.mean(y_data), height , height, np.mean(y_data), height * 3, height]

    try:
        # optimize and in the end you will have 6 coeff (3 for each gaussian)
        coeff, var_matrix = curve_fit(gauss2, x_data, y_data, p0=p0)
    except RuntimeError:
        print('Oh oh, could not fit')
        return []

    means = [round(coeff[1]), round(coeff[4])]

    if any([x < 0 or x > len(brightnesses) for x in means]):
        print(f'Error in brightness: Means = {means} length of brightness list = {len(brightnesses)}')
        return []

    # you can plot each gaussian separately using
    pg1 = np.zeros_like(p0)
    pg1[0:3] = coeff[0:3]
    pg2 =np.zeros_like(p0)
    pg2[0:3] = coeff[3:]

    g1 = gauss2(x_data, *pg1)
    g2 = gauss2(x_data, *pg2)

    if plots >= 1:

        plt.figure()
        plt.plot(x_data, y_data, label='Data')
        plt.plot(x_data, g1, label='Fit1')
        plt.plot(x_data, g2, label='Fit2')

        plt.scatter(means, y_data[means], c='red')

        plt.title('brightness dist & underlying dists')
        plt.ylabel('brightness')
        plt.xlabel('slice')
        plt.legend(loc='upper right')

        plt.show()
        # plt.savefig(r'.\brightnesses_gaussian_fit.png')

        return means, g1, g2

    return means, g1, g2


def get_neuron_lengths_dict(arr):
    """
    Gets the length of each neuron/mask across Z.
    IMPORTANT: A stitched array  unique and consecutive values only! is assumed.

    Parameters
    ----------
    arr : numpy array
        Data array (ZXY) of masks

    Returns
    -------
    lengths : dict
        Dictionary containing the lengths of each found neurons
        {neuron # (int) : length(int)}
    """
    # TODO add part to keep track, if a found ID has gaps within Z-planes to avoid wrong lengths

    neurons = np.unique(arr)
    lengths = defaultdict(int)

    for neuron in neurons:
        if neuron == 0:
            continue
        z_count = 0

        for plane in range(len(arr)):
            if neuron in arr[plane]:
                z_count += 1

        lengths[neuron] = z_count

    return lengths


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


def create_3d_array_from_tiff(img_path: str, flyback_flag=0):
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
        return []

    if len(img_list) >= 2:
        print('There is more than 1 TIFF file in the folder! Are you sure, that is the correct folder?')
        print(img_list)
        print(f'Will use {img_list[0]}')

    with tiff.TiffFile(img_list[0]) as my_tiff:
        size = (len(my_tiff.pages), *my_tiff.pages[0].asarray().shape)  # '*' to unpack tuple right away
        img_3d_array = np.zeros(size)

        for p, page in enumerate(my_tiff.pages):
            img_3d_array[p] = page.asarray()

    # skipping first plane if flyback == 1
    if flyback_flag:
        img_3d_array = img_3d_array[1:, ...]

    return img_3d_array


def array_dispatcher(vol_path, align=False, remove_flyback=True):
    """
    Checks, whether the data is .tif or .npy and creates a 3D array accordingly.

    Parameters
    ----------
    vol_path : str
        Path of volume
    align : bool
        If True, raw TIFF file is assumed and will be pre-aligned.
    remove_flyback : bool
        If True, the first slice will be removed AFTER pre-aligning or segmenting.
    Returns
    -------
        3D numpy array
    """
    # check, for folder content
    files = [os.path.join(vol_path, f.name) for f in os.scandir(vol_path) if f.is_file()]

    if align and '.tif' in files[0]:
        raw_array_3d = prealign.rigid_prealignment(files[0])
    elif align and '.tif' not in files[0]:
        print(f'Unexpected problem! {files[0]} is not a TIF file, but pre-alignment was ON.\nExiting.')
    else:
        if '.tif' in files[0]:
            raw_array_3d = create_3d_array_from_tiff(vol_path)
        elif '.npy' in files[0]:
            raw_array_3d = create_3d_array(vol_path)
        else:
            print('Neither TIFF nor npy file in folder! Exiting!')
            pass

    # TODO change segmentation method/algo depending on the accuracy results
    # TODO segment incoming volume, not a file!
    print(f'... Segmentation start (in dispatcher). File: {files[0]}')
    seg_array_3d = sd.segment_with_stardist(files[0])

    if remove_flyback:
        raw_array_3d = raw_array_3d[1:]
        seg_array_3d = seg_array_3d[1:]

    return raw_array_3d, seg_array_3d


