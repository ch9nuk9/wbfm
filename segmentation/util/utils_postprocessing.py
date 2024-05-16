"""
Postprocessing functions for segmentation pipeline
"""
import concurrent
import logging
from typing import List, Tuple, Union
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy
import zarr
from skimage.morphology import erosion
from skimage.segmentation import find_boundaries
from tqdm.auto import tqdm
import skimage
from scipy.optimize import curve_fit
from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.general.utils_networkx import calc_bipartite_from_candidates
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.tracklets.utils_tracklets import build_tracklets_dfs
from scipy.signal import find_peaks
from skimage.measure import regionprops

from segmentation.util.util_curve_fitting import calculate_multi_gaussian_fits, get_best_model_using_aicc, \
    calc_split_point_from_gaussians, plot_gaussians, _plot_just_data


def remove_large_areas(arr: np.ndarray, threshold=1000, verbose=0):
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
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    arr : 2D or 3D numpy array
        array with removed areas. Same shape as input
    """
    if verbose >= 1:
        print('Removing large areas in all planes')

    new_arr = arr.copy()
    if len(arr.shape) > 2:
        for i, plane in enumerate(arr):
            for u in np.unique(plane):
                if u == 0:
                    # Background
                    continue
                mask = plane == u
                if np.count_nonzero(mask) >= threshold:
                    plane[mask] = 0
            new_arr[i] = plane
    else:
        uniq = np.unique(arr)
        for u in uniq:
            if np.count_nonzero(arr == u) >= threshold:
                new_arr = np.where(arr == u, 0, arr)
    return new_arr


def remove_dim_slices(masks, img_volume, thresh_factor=1.1, verbose=0):
    threshold = thresh_factor * np.mean(img_volume)

    for vol_slice, mask_slice in zip(img_volume, masks):
        all_neurons = np.unique(mask_slice)
        for neuron in all_neurons:
            if neuron == 0:
                continue
            mask = mask_slice == neuron
            brightness = np.mean(vol_slice[mask])
            if brightness < threshold:
                mask_slice[mask] = 0

    return masks


def bipartite_stitching(array_3d: np.ndarray, num_slices=0, verbose=0) -> Tuple[np.ndarray, tuple]:
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
        flag for print statements. Increasing by 1, increase depth by 1

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
    if verbose >= 1:
        print(f'Bipartite stitching. Input array shape: {array_3d.shape}')

    num_slices = len(array_3d)

    # Initialize output matrix
    all_matches = {}  # Indexed by which pair of z slices

    for i_slice in range(num_slices):

        this_slice = array_3d[i_slice]
        if i_slice < num_slices - 1:
            next_slice = array_3d[i_slice + 1]
            match_key = (i_slice, i_slice + 1)

            this_slice_candidates = create_matches_list(this_slice, next_slice)
            if len(this_slice_candidates) == 0:
                continue

            # Bipartite matching after creating overlap list for all neurons on slice
            bp_matches = calc_bipartite_from_candidates(this_slice_candidates)[0]
            all_matches[match_key] = bp_matches

    clust_df = build_tracklets_dfs(all_matches)

    # renaming all found neurons in array; in a sorted manner
    sorted_stitched_array = rename_stitched_array(array_3d, clust_df)
    out = (clust_df, all_matches)

    return sorted_stitched_array, out


def stitch_via_watershed(seg_dat, red_dat, sigma=1, verbose=0):
    """
    New method for creating 3d objects via watershed

    Parameters
    ----------
    seg_dat
    red_dat
    sigma

    Returns
    -------

    """

    # Filter the brightness
    filtered_red = skimage.filters.gaussian(red_dat, sigma=sigma)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    if verbose >= 1:
        print("Applying distance transform...")
    distance = scipy.ndimage.distance_transform_edt(seg_dat.astype(bool))
    distance_times_brightness = np.multiply(distance, filtered_red)
    coords = skimage.feature.peak_local_max(distance_times_brightness, footprint=np.ones((5, 11, 11)))
    # Can be used if the labels are basically correct and need to be refined, but not if they are just boolean
    #     coords = peak_local_max(distance_times_brightness, footprint=np.ones((5, 11, 11)), labels=seg_dat, num_peaks_per_label=3)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    if verbose >= 1:
        print("Resegmenting using watershed...")
    labels = skimage.segmentation.watershed(-distance, markers, mask=seg_dat)
    if verbose >= 1:
        print("Finished distance transform mask calculation")

    return labels


def create_matches_list(slice_1, slice_2, min_overlap=5, verbose=0) -> list:
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
            if overlap_area > min_overlap:
                bip_inter.append([int(this_neuron), int(overlap_neuron), int(overlap_area)])

        bip_list.extend(bip_inter)

    return bip_list


def rename_stitched_array(arr, df, verbose=0):
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
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    sorted array : 3D numpy array
        3D array of masks with unique values across Z
    """
    if verbose >= 1:
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

    neurons_in_plane_all = [set(np.unique(plane)) for plane in arr]

    for neuron in neurons:
        if neuron == 0:
            continue
        z_count = 0

        for set_of_neurons in neurons_in_plane_all:
            if neuron in set_of_neurons:
                z_count += 1

        lengths[int(neuron)] = z_count

    return lengths


def OLD_calc_brightness(original_array, stitched_masks, neuron_lengths=None, verbose=0):
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
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    brightness_dict : dict
        contains a list of average brightness values per plane for each neuron
        {neuron #: [average brightnesses per plane]}
    brightness_planes : dict
        contains the global Z plane index for each neuron
        {neuron #: [global Z planes]}

    """
    if verbose >= 1:
        print('Start with brightness calculations')
    # add default dict
    brightness_dict = defaultdict(list)
    brightness_planes = defaultdict(list)

    # loop over the actual data and calculate average brightness per neuron per slice/mask
    if neuron_lengths is None:
        neuron_lengths = list(np.unique(stitched_masks)[1:])
    else:
        neuron_lengths = neuron_lengths.keys()

    for neuron in neuron_lengths:
        current_list = list()
        planes_list = list()

        for i_slice, slice in enumerate(stitched_masks):
            # get the mask
            if int(neuron) in slice:
                this_mask = slice == int(neuron)

                # get the average brightness for that mask
                # TODO: integrate instead of average
                current_brightness = int(np.nanmean(original_array[i_slice, this_mask]))

                # extend the brightness dict
                if current_brightness is not np.nan:
                    current_list.append(current_brightness)
                    planes_list.append(i_slice)
                else:
                    if verbose >= 1:
                        print(f'NaN in neuron {neuron} slice {i_slice}')

                # TODO can be optimized by breaking the loop after not finding anything
                #  (but beginning slice needs to be figured out)

        # add list of brightnesses to dict
        brightness_dict[neuron].extend(current_list)
        brightness_planes[neuron].extend(planes_list)
    if verbose >= 1:
        print(f'Done with  calculating brightnesses')

    return brightness_dict, brightness_planes


def calc_brightness(original_array: np.ndarray, stitched_masks, verbose=0) -> Tuple[dict, dict, dict]:
    """
    Calculates the average brightness of each mask per plane in an array given the original image
    and the mask lengths.

    Parameters
    ----------
    original_array : 3D numpy array
        original image array
    stitched_masks : 3D numpy array
        array with stitched (=unique and consecutive values per neuron) neuron masks
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    brightness_dict : dict
        contains a list of average brightness values per plane for each neuron
        {neuron #: [average brightnesses per plane]}
    brightness_planes : dict
        contains the global Z plane index for each neuron
        {neuron #: [global Z planes]}

    """
    brightness_dict = defaultdict(list)
    centroid_dict = defaultdict(list)
    brightness_planes = defaultdict(list)

    for z in range(stitched_masks.shape[0]):
        props = regionprops(np.array(stitched_masks[z, ...]), intensity_image=np.array(original_array[z, ...]))
        for p in props:
            label = p.label
            centroid_dict[label].append(p.centroid)
            img = p.intensity_image  # Includes a bounding box with 0s
            img = img[img > 0]
            brightness_dict[label].append(int(np.nanmean(img)))
            # brightness_dict[label].append(int(p.intensity_mean))  # Works in later skimage
            brightness_planes[label].append(z)
    if verbose >= 1:
        print(f'Done with  calculating brightnesses')

    return brightness_dict, brightness_planes, centroid_dict


def calc_split_point_via_brightnesses(brightnesses, min_separation,
                                      num_gaussians=2,
                                      min_height=5,
                                      plots=0, verbose=0,
                                      to_save_plot=0,
                                      force_split=False,
                                      return_all=False) -> Tuple[Union[int, None], List[str]]:
    """
    New function that mostly removes parameters in favor of an AICC calculation to determine if it is one or
    two gaussians

    NOTE: mostly removes the hard threshold I had before; AICC is more stable

    Parameters
    ----------
    brightnesses
    min_separation
    num_gaussians
    min_height
    plots
    verbose
    to_save_plot
    return_all

    Returns
    -------

    """

    if len(brightnesses) < 7:
        # Note: a length of 7 will have an infinite aicc penalty and will never be split, but a user can force it
        if plots:
            _plot_just_data(list(range(len(brightnesses))), brightnesses)
        return None, ["Too short"]

    list_of_models = calculate_multi_gaussian_fits(brightnesses, background=14)
    if verbose >= 3:
        print(list_of_models)
    if force_split:
        i_best = 1
        all_aicc=[]
    else:
        i_best, all_aicc = get_best_model_using_aicc(list_of_models)
    model = list_of_models[i_best]

    if i_best == 0:
        split_point = None
        explanation = [f"Single gaussian (aicc): {all_aicc}"]
    elif i_best == 1:
        split_point = calc_split_point_from_gaussians(model)
        explanation = [f"Two gaussians (aicc): {all_aicc}"]
    elif i_best == 2:
        # TODO: will be 2 split points
        split_point1 = calc_split_point_from_gaussians(model)
        split_point2 = calc_split_point_from_gaussians(model, prefix1='g2_', prefix2='g3_')
        if split_point2 is not None:
            split_point = [split_point1, split_point2]
        else:
            split_point = split_point1
        explanation = [f"Three gaussians (aicc): {all_aicc}"]
    else:
        raise NotImplementedError
    if plots:
        plot_gaussians(model, split_point)

    return split_point, explanation


def OLD_calc_split_point_via_brightnesses(brightnesses, min_separation,
                                      num_gaussians=2,
                                      min_height=5,
                                      plots=0, verbose=0,
                                      to_save_plot=0,
                                      return_all=False) -> Tuple[int, List[str]]:
    """
    calculates the means of 2 gaussians underlying the neuron brightness distributions.
    It tries to match exactly 2 gaussians onto the brightness distribution of a tentative neuron.

    Parameters
    ----------
    num_gaussians : int
        Number of gaussians to fit [2 or 3]
    brightnesses : list
        List containing average brightness values of a tentative neuron
    min_separation : int
        Minimum separation between the peaks for them to count as "real"
    min_height : int
        Used in postprocessing to make sure one height isn't extremely tiny
    plots :
        flag for plotting
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    split_point : int
        The split point in the same coordinate system as brightnesses

    OLD:
    -------
    means : list
        list of means of 2 underlying gaussians, IF they could be fitted
    g1, g2 : list
        list containing the values of the 2 gaussians, IF the y could be fitted
    """

    y_data = np.array(brightnesses, dtype=float)
    x_data = np.array(np.arange(len(y_data)), dtype=int)

    explanation = ["Successfully split"]

    if len(y_data) < 8:
        explanation[0] = f"Can't split a neuron that is too short ({len(y_data)}); aborting"
        logging.warning(explanation[0])
        if plots >= 1:
            fig = _plot_just_data(x_data, y_data)
            plt.title(explanation[0])
            plt.show()
        return None, explanation

    # Define model function to be used to fit to the data above:
    # Adapt it to as many gaussians you may want
    # by copying the function with different A2,mu2,sigma2 parameters

    def gauss1(x, *p):
        A1, mu1, sigma1 = p
        return A1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2))

    def gauss2(x, *p):
        return gauss1(x, *p[1:4]) + gauss1(x, *p[4:]) + p[0]

    # def gauss2(x, *p):
    #     A1, mu1, sigma1, A2, mu2, sigma2 = p
    #     return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

    def gauss3(x, *p):
        raise NotImplementedError
        return gauss1(x, *p[:3]) + gauss1(x, *p[3:6]) + gauss1(x, *p[6:])

    p0 = get_initial_gaussian_peaks(min_separation, num_gaussians, y_data)

    if num_gaussians == 2:
        fit_func = gauss2
    elif num_gaussians == 3:
        if verbose >= 1:
            print("Attempting to fit 3 gaussians")
        fit_func = gauss3

    try:
        # optimize and in the end you will have 6 coeff (3 for each gaussian)
        coeff, var_matrix = curve_fit(fit_func, x_data, y_data, p0=p0)
    except RuntimeError:
        if verbose >= 1:
            print('Oh oh, could not fit')
        if plots >= 1:
            fig = _plot_just_data(x_data, y_data)
            explanation = "Runtime error when fitting 2 gaussians"
            plt.title(explanation)
            plt.show()
        return None, explanation

    if num_gaussians == 2:
        peaks_of_gaussians = [round(coeff[2]), round(coeff[5])]
    elif num_gaussians == 3:
        peaks_of_gaussians = [round(coeff[2]), round(coeff[5]), round(coeff[8])]
    peaks_of_gaussians, peak_explanation = sanity_checks_on_peaks(brightnesses, peaks_of_gaussians, verbose, y_data,
                                                                  min_separation)
    heights_of_gaussians = [round(coeff[1]), round(coeff[4])]
    peaks_of_gaussians, height_explanation = sanity_checks_on_heights(heights_of_gaussians, peaks_of_gaussians, min_height, verbose)

    explanation.extend([peak_explanation, height_explanation])

    if num_gaussians == 2:
        split_point = calc_split_point_from_gaussians(peaks_of_gaussians, y_data)
    elif num_gaussians == 3:
        split_point1 = calc_split_point_from_gaussians(peaks_of_gaussians[:2], y_data)
        split_point2 = calc_split_point_from_gaussians(peaks_of_gaussians[1:], y_data)
        split_point = [split_point1, split_point2]
    else:
        raise NotImplementedError

    if plots >= 1 and num_gaussians == 2:
        g1, g2, y_data, fig = _plot_gaussians(coeff, gauss1, peaks_of_gaussians, x_data, y_data, split_point,
                                              num_gaussians=2)
        if to_save_plot:
            fname = f"split_at_{split_point}_with_{num_gaussians}gaussian_fit.png"
            print(f"Saving figure at : {fname}")
            plt.savefig(fname)
        else:
            print("Plotted but did not save image")
        # if return_all:
        #     return split_point, peaks_of_gaussians, y_data, g1, g2, coeff, p0, explanation
        # elif return_all:
        #     g1 = gauss1(x_data, *coeff[0:3])
        #     return split_point, None, g1, None

    return split_point, explanation


def get_initial_gaussian_peaks(min_separation, num_gaussians, y_data, background_val=15):
    # p0 is the initial guess for the fitting coefficients
    # initialize the two gaussians differently so the optimization algorithm works better
    sigma = min_separation / 2
    peaks, _ = find_peaks(y_data, distance=sigma)
    if num_gaussians == 2:
        if len(peaks) == 2:
            peak0, peak1 = peaks
        else:
            peak0 = len(y_data) / 4.0
            peak1 = peak0 * 3
        p0 = [np.mean(y_data), peak0, sigma, np.mean(y_data), peak1, sigma]
    elif num_gaussians == 3:
        if len(peaks) == 3:
            peak0, peak1, peak2 = peaks
        else:
            peak0 = len(y_data) / 5.0
            peak1 = peak0 * 2
            peak2 = peak0 * 4
        p0 = [np.mean(y_data), peak0, sigma,
              np.mean(y_data), peak1, sigma,
              np.mean(y_data), peak2, sigma]
    else:
        raise NotImplementedError
    # Background value
    p0.insert(0, background_val)
    return p0


def sanity_checks_on_peaks(brightnesses, peaks_of_gaussians: list, verbose, y_data, min_separation: int):
    explanation = "Peak locations are fine"
    if any([x < 0 or x > len(y_data) for x in peaks_of_gaussians]):
        # Positions outside the neuron
        explanation = "Impossible location; returning None"
        if verbose >= 1:
            print(f'Error in brightness: Means = {peaks_of_gaussians} length of brightness list = {len(brightnesses)}')
            print(explanation)
        return None, explanation
    elif any(np.abs(np.diff(peaks_of_gaussians)) < min_separation):
        explanation = f'Peaks too close, aborting: Means = {peaks_of_gaussians}'
        if verbose >= 1:
            print(explanation)
        return None, explanation
    else:
        return peaks_of_gaussians, explanation


def sanity_checks_on_heights(heights_of_gaussians, peaks_of_gaussians, min_height, verbose):
    explanation = "Heights are fine"
    if any(x < min_height for x in heights_of_gaussians):
        # Tiny (non-physical) heights
        explanation = f'Non-physical heights, returning None: {heights_of_gaussians}'
        if verbose >= 1:
            print(explanation)
        return None, explanation
    else:
        return peaks_of_gaussians, explanation


def _plot_gaussians(coeff, gauss1, peaks_of_gaussians, x_data, y_data, split_point, num_gaussians):
    background = coeff[0]
    y_data -= background
    coeff = coeff[1:]
    g1 = gauss1(x_data, *coeff[:3])
    if num_gaussians > 1:
        g2 = gauss1(x_data, *coeff[3:])
    else:
        g2 = None
    fig = _plot_just_data(x_data, y_data)
    plt.plot(x_data, g1, label='Fit1')
    plt.plot([split_point, split_point], [0, np.max(y_data)], 'k', label="Split point")
    if num_gaussians > 1:
        plt.plot(x_data, g2, label='Fit2')
    if peaks_of_gaussians is not None:
        for peak in peaks_of_gaussians:
            if peak >= len(y_data):
                logging.warning(f"Peak {peak} was off the plot!")
                continue
            plt.scatter(peak, y_data[peak], c='red')
    return g1, g2, y_data, fig


def split_long_neurons(mask_array,
                       neuron_lengths: dict,
                       neuron_brightnesses: dict,
                       neuron_centroids: dict,
                       global_current_neuron,
                       maximum_length,
                       neuron_z_planes: dict,
                       min_separation: int,
                       also_split_using_centroids: bool,
                       verbose=0):
    """
    Splits neuron, which are too long (to our understanding) into 2 parts. The split-point is the midpoint between
    2 gaussians, that have been fit onto a plot of the average brightness (per slice) of that neuron. The second half
    of the split neuron gets a new ID and will be appended.

    Parameters
    ----------
    mask_array : 3D numpy array
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
    maximum_length : int or list
        Threshold for neuron length; if 2 values, then the second threshold will trigger an attempt to fit 3 gaussians
    neuron_z_planes : dict(list)
        Contains the Z-planes corresponding to the brightness values of each neuron
        neuron_ID = 1
        neuron_z_planes[neuron_ID] == [12, 13, 14]
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

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

    if np.isscalar(maximum_length):
        length_for_2_gaussians = maximum_length
        length_for_3_gaussians = np.inf
    elif len(maximum_length) == 2:
        length_for_2_gaussians, length_for_3_gaussians = maximum_length
    else:
        raise NotImplementedError

    # iterate over neuron lengths dict, and if z >= 12, try to split it
    new_neuron_lengths = {}
    for neuron_id, neuron_len in neuron_lengths.items():
        if neuron_len > length_for_2_gaussians:
            num_gaussians = 2
            if neuron_len > length_for_3_gaussians:
                num_gaussians = 3

            try:
                x_split_local_coord, explanation = calc_split_point_via_brightnesses(neuron_brightnesses[neuron_id],
                                                                                     min_separation,
                                                                                     num_gaussians=num_gaussians,
                                                                                     verbose=verbose - 1)
                if also_split_using_centroids and x_split_local_coord is None:
                    grads = calc_centroid_difference(neuron_centroids[neuron_id])
                    x_split_local_coord = get_split_point_from_centroids(grads, threshold=4)
                    if not (1 <= x_split_local_coord <= len(grads)-1):
                        # Would make too short of a neuron
                        x_split_local_coord = None

            except (ValueError, TypeError) as err:
                if verbose >= 1:
                    print(f'Error while splitting neuron {neuron_id}: Could not fit 2 Gaussians! Will continue.')
                    print(err)
                continue

            # if neuron can be split
            if x_split_local_coord is not None:
                if np.isscalar(x_split_local_coord):
                    x_split_local_coord = [x_split_local_coord]
                for i in x_split_local_coord:
                    # print(x_split_local_coord, explanation)
                    global_current_neuron = split_neuron_and_update_dicts(global_current_neuron, mask_array,
                                                                          neuron_brightnesses, neuron_id,
                                                                          neuron_lengths,
                                                                          neuron_z_planes, new_neuron_lengths,
                                                                          i)
                if verbose >= 1:
                    print(f"Successfully Fit 2 gaussians to neuron {neuron_id} with split: {x_split_local_coord}")
                    print(f"Additional neuron ID: {global_current_neuron}")

    neuron_lengths.update(new_neuron_lengths)

    return mask_array, neuron_lengths, neuron_brightnesses, global_current_neuron, neuron_z_planes


def split_neuron_and_update_dicts(global_current_neuron, mask_array, neuron_brightnesses, neuron_id, neuron_lengths,
                                  neuron_z_planes, new_neuron_lengths, x_split_local_coord):
    # create new entry
    global_current_neuron += 1
    # try:
    new_neuron_lengths[global_current_neuron] = neuron_lengths[neuron_id] - x_split_local_coord - 1
    # except TypeError as err:
    #     print(neuron_lengths[neuron_id])
    #     print(x_split_local_coord)
    #     print(neuron_id)
    #     raise err
    # update neuron lengths and brightnesses entries; 0-x_split = neuron 1
    new_neuron_lengths[neuron_id] = x_split_local_coord + 1
    # Convert the length to the z indices of the full mask, not local to the neurons
    x_split_global_coord = x_split_local_coord + neuron_z_planes[neuron_id][0]
    # update mask array with new mask IDs
    for plane in mask_array[x_split_global_coord+1:]:
        if neuron_id in plane:
            inter_plane = plane == neuron_id
            plane[inter_plane] = global_current_neuron
    # update brightnesses and brightness-planes dicts
    neuron_brightnesses[global_current_neuron] = neuron_brightnesses[neuron_id][x_split_local_coord + 1:]
    neuron_brightnesses[neuron_id] = neuron_brightnesses[neuron_id][:x_split_local_coord + 1]
    neuron_z_planes[global_current_neuron] = neuron_z_planes[neuron_id][x_split_local_coord + 1:]
    neuron_z_planes[neuron_id] = neuron_z_planes[neuron_id][:x_split_local_coord + 1]
    return global_current_neuron


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

    rm_list = list()
    for key, value in neuron_lengths.items():
        if value < length_cutoff:
            cull = array == key
            array[cull] = 0

            rm_list.append(key)

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

    masks[:, :border, :] = 0
    masks[:, (x_sz - border):, :] = 0
    masks[:, :, :border] = 0
    masks[:, :, (y_sz - border):] = 0

    return masks


##
## For interactive post-processing
##
def split_neuron_interactive(full_mask, red_volume, i_target,
                             min_separation=2,
                             which_neuron_keeps_original='top',
                             method='gaussian',
                             verbose=0,
                             **kwargs):
    """
    A user will decide that "i_target" definitely needs to be split, so this will try all possible methods to do so

    Parameters
    ----------
    min_separation
    red_volume
    full_mask
    i_target
    which_neuron_keeps_original

    Returns
    -------

    """

    # Calculate the brightness per plane (used in all methods)
    brightness_per_plane, individual_plane_masks, planes_where_neuron_exists = get_brightness_per_plane(full_mask,
                                                                                                        i_target,
                                                                                                        red_volume)

    if verbose > 2:
        opt = {'to_save_plot': False, 'plots': True}
    else:
        opt = {'to_save_plot': False, 'plots': False}

    if method.lower() == 'gaussian':
        # Method 1: Gaussian fitting
        x_split_local_coord, explanation = calc_split_point_via_brightnesses(brightness_per_plane,
                                                                             min_separation=min_separation,
                                                                             force_split=True,
                                                                             **opt)
        # Check for success
        if x_split_local_coord is None:
            logging.warning("Could not split using Gaussian method")
            print(explanation)
        elif verbose >= 1:
            print(f"Split point is {x_split_local_coord} using gaussian method")
            print(explanation)

            # Actual split
            # TODO: Should the split ind go in top neuron?
            new_full_mask = update_neuron_within_full_mask(full_mask, individual_plane_masks,
                                                           planes_where_neuron_exists,
                                                           which_neuron_keeps_original, x_split_local_coord)
            return new_full_mask

    elif method.lower() == 'spatial_dot_product':
        # Method 2: dot product between planes and refit gaussian
        planes_with_single_neuron = []
        for i, plane in enumerate(full_mask):
            if i_target in plane:
                plane_mask = plane == i_target
                plane_red = red_volume[i]
                planes_with_single_neuron.append(np.where(plane_mask, plane_red, 0))
        all_dots = []
        for i in range(len(planes_with_single_neuron) - 1):
            all_dots.append(np.linalg.norm(planes_with_single_neuron[i] * planes_with_single_neuron[i + 1]))

        # Fit gaussians
        x_split_local_coord, explanation = calc_split_point_via_brightnesses(all_dots,
                                                                             min_separation=min_separation,
                                                                             force_split=True,
                                                                             **opt)

        # Check for success
        if x_split_local_coord is None:
            logging.warning("Could not split using Gaussian method")
            print(explanation)
        else:
            if verbose >= 1:
                x_split_local_coord += 0.5
                print(f"Split point is {x_split_local_coord} using dot-product gaussian method")
                print(explanation)

            new_full_mask = update_neuron_within_full_mask(full_mask, individual_plane_masks,
                                                           planes_where_neuron_exists,
                                                           which_neuron_keeps_original, x_split_local_coord)

            return new_full_mask

    elif method.lower() == 'manual':
        x_split_local_coord = kwargs['x_split_local_coord']
        new_full_mask = update_neuron_within_full_mask(full_mask, individual_plane_masks, planes_where_neuron_exists,
                                                       which_neuron_keeps_original, x_split_local_coord)

        return new_full_mask

    else:
        raise NotImplementedError(f"Unrecognized method {method}")

    # Method 3: centroid discontinuity?

    return None


def get_brightness_per_plane(full_mask, i_target, red_volume):
    brightness_per_plane = []
    planes_where_neuron_exists = []
    individual_plane_masks = []
    for i, plane in enumerate(full_mask):
        if i_target in plane:
            plane_mask = plane == i_target
            plane_red = red_volume[i]
            brightness_per_plane.append(np.sum(plane_red[plane_mask]))
            planes_where_neuron_exists.append(i)
            individual_plane_masks.append(plane_mask)
    assert len(brightness_per_plane) > 0, f"Neuron {i_target} not found!"
    return brightness_per_plane, individual_plane_masks, planes_where_neuron_exists


def update_neuron_within_full_mask(full_mask, individual_plane_masks, planes_where_neuron_exists,
                                   which_neuron_keeps_original, x_split_local_coord):
    new_neuron_id = np.max(full_mask) + 1
    new_full_mask = full_mask.copy()
    for i_local, i_z in enumerate(planes_where_neuron_exists):
        if i_local > x_split_local_coord and which_neuron_keeps_original.lower() == 'top':
            this_plane = new_full_mask[i_z]
            this_plane[individual_plane_masks[i_local]] = new_neuron_id
            new_full_mask[i_z] = this_plane
        elif i_local <= x_split_local_coord and which_neuron_keeps_original.lower() == 'bottom':
            this_plane = new_full_mask[i_z]
            this_plane[individual_plane_masks[i_local]] = new_neuron_id
            new_full_mask[i_z] = this_plane
    return new_full_mask


def get_centroid_diff_for_single_neuron(seg, red, i):
    this_neuron = np.where(seg == i, red, 0)
    this_mask = np.where(seg == i, seg, 0)

    xy = []
    for z in range(seg.shape[0]):
        b = np.mean(this_neuron[z, ...])
        if b > 0:
            props = regionprops(this_mask[z, ...])
            xy.append(props[0].centroid)
    sum_of_grads = calc_centroid_difference(xy)

    return sum_of_grads


def calc_centroid_difference(xy):
    xy = np.array(xy)
    grad_x = np.diff(xy[:, 0])
    grad_x -= np.mean(grad_x)
    grad_y = np.diff(xy[:, 1])
    grad_y -= np.mean(grad_y)
    sum_of_grads = grad_x ** 2 + grad_y ** 2
    return sum_of_grads


def get_split_point_from_centroids(sum_of_grads, threshold=4):
    if np.max(sum_of_grads) > threshold:
        ind = np.argmax(sum_of_grads)
        return ind
    else:
        return None


def zero_out_borders_using_config(project_config: ModularProjectConfig):
    """
    Modifies the segmentation to remove all boundaries (touching masks)

    Same effect as setting 'zero_out_borders' and 'do_full_erosion' in the initial segmentation config file

    Parameters
    ----------
    project_config

    Returns
    -------

    """

    segment_cfg = project_config.get_segmentation_config()
    old_seg_fname = segment_cfg.resolve_relative_path_from_config('output_masks')
    old_masks = zarr_reader_folder_or_zipstore(old_seg_fname)
    num_frames = old_masks.shape[0]
    do_full_erosion = segment_cfg.config['segmentation_params'].get('full_erosion', False)

    new_seg_fname = get_sequential_filename(old_seg_fname)
    new_masks = zarr.zeros_like(old_masks, store=new_seg_fname)
    # new_masks = zarr.open(new_seg_fname, mode='a',
    #                       shape=old_masks.shape, chunks=old_masks.chunks, dtype=np.uint16,
    #                       fill_value=0,
    #                       synchronizer=zarr.ThreadSynchronizer())

    with tqdm(total=num_frames) as pbar:
        def parallel_func(i):
            labels = old_masks[i].copy()
            if do_full_erosion:
                labels = erosion(labels)
            labels_bd = find_boundaries(labels, connectivity=2, mode='outer', background=0)
            this_mask = labels
            this_mask[labels_bd] = 0
            new_masks[i] = this_mask

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(parallel_func, i): i for i in range(num_frames)}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    updates = {'output_masks': segment_cfg.unresolve_absolute_path(new_seg_fname)}
    segment_cfg.config.update(updates)

    segment_cfg.update_self_on_disk()
    segment_cfg.logger.info(f'Done with segmentation postprocessing! Mask data saved at {new_seg_fname}')


def create_crop_masks_using_config(project_config: ModularProjectConfig, target_sz: np.ndarray):
    """
    Modifies the segmentation to be only a tiny mask around the centroid.
    Note that the centroid is calculated from the original mask

    Parameters
    ----------
    project_config
    target_sz

    Returns
    -------

    """

    segment_cfg = project_config.get_segmentation_config()
    old_seg_fname = segment_cfg.resolve_relative_path_from_config('output_masks')
    old_masks = zarr_reader_folder_or_zipstore(old_seg_fname)
    num_frames = old_masks.shape[0]
    dz, dx, dy = (np.array(target_sz) / 2.0).astype(int)

    new_seg_fname = get_sequential_filename(old_seg_fname)
    new_masks = zarr.open(new_seg_fname, chunks=old_masks.chunks, shape=old_masks.shape, fill_value=0, dtype=int)
    # new_masks = zarr.zeros_like(old_masks, store=new_seg_fname)

    with tqdm(total=num_frames) as pbar:
        def parallel_func(i):
            labels = old_masks[i].copy()
            props = regionprops(labels)
            for p in tqdm(props, leave=False):
                # Get centroids (just recalculate, unweighted)
                label = int(p.label)
                centroid = p.centroid

                # Get bbox of the smaller size
                z0, z1 = int(centroid[0] - dz), int(centroid[0] + dz)
                x0, x1 = int(centroid[1] - dx), int(centroid[1] + dx)
                y0, y1 = int(centroid[2] - dy), int(centroid[2] + dy)

                # Apply
                new_masks[i, z0:z1, x0:x1, y0:y1] = label

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(parallel_func, i): i for i in range(num_frames)}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    updates = {'output_masks': segment_cfg.unresolve_absolute_path(new_seg_fname)}
    segment_cfg.config.update(updates)

    segment_cfg.update_self_on_disk()
    segment_cfg.logger.info(f'Done with segmentation postprocessing! Mask data saved at {new_seg_fname}')
