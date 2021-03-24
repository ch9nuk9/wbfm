"""
Postprocessing functions for segmentation pipeline
"""
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_matches
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_from_matches


def remove_large_areas(arr, threshold=1000):
    """
    Iterates overs planes of array and removes areas, which are larger than 'threshold'.
    May take in a 2D or 3D array.

    Parameters
    ----------
    arr : 2D or 3D numpy array
        Array of segmented masks. Can be used for
    threshold : int
        Threshold for maximum size of a patch in a plane.
        Default was eyeballed by looking at the distribution of areas within mis-segmented planes.

    Returns
    -------
    arr : 2D or 3D numpy array
        array with removed areas. Same shape as input
    """

    if len(arr.shape) > 2:
        for i, plane in enumerate(arr):
            uniq = np.unique(plane)

            for u in uniq:
                if np.count_nonzero(plane == u) >= threshold:
                    plane = np.where(plane == u, 0, plane)

            arr[i] = plane
    else:
        uniq = np.unique(arr)
        for u in uniq:
            if np.count_nonzero(arr == u) >= threshold:
                arr = np.where(arr == u, 0, arr)
    return arr


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
            bp_matches = sorted(calc_bipartite_matches(this_slice_candidates))

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


def remove_border(masks, border=100):
    """
    Puts image values, which are 'border' pixels (default= 100 px) away from any edge, to 0.
    Reason: segmentation produces many edge artefacts, which are ~neuron sized in prealigned volumes.

    Parameters
    ----------
    masks : 3D numpy array
        Array of stitched masks. Presumably with artefacts.
    border : int
        Distance from edges until which values will be zeroed.
    Returns
    -------
    masks : 3D numpy array
        Array with removed edge values. Should contain little to no edge artefacts anymore.
    """
    _, x_sz, y_sz = masks.shape

    masks[:, :border, :] = 0.0
    masks[:, (x_sz - border):, :] = 0.0
    masks[:, :, :border] = 0.0
    masks[:, :, (y_sz - border):] = 0.0

    return masks
