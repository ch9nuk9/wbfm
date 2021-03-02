import numpy as np
import os
from collections import defaultdict
import segmentation.util.overlap as ol
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

# bipartite matching modules/functions
import networkx as nx
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import get_node_name
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import unpack_node_name
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import calc_bipartite_matches


def create_2d_masks_gt():
    # creates separate 2d masks of the annotated ground truth
    path = r'C:\Segmentation_working_area\ground_truth\one_volume_seg.npy'
    arr = np.load(path, allow_pickle=True).item()
    masks = arr['masks']

    sv_path = r'C:\Segmentation_working_area\ground_truth\gt_masks_npy'
    c = 0
    for m in masks[1:]:
        np.save(os.path.join(sv_path, 'gt_mask_' + str(c)), m, allow_pickle=True)
        print(os.path.join(sv_path, 'gt_mask_' + str(c)))
        c += 1

# create_2d_masks_gt()


# refactor into using 3d arrays instead of 2d
def gmm():
    # create a data -> brightness from a single neuron
    # data needed: original 3d array, stitched array of choice, neuron lengths

    img_data_path = r'C:\Segmentation_working_area\test_volume'
    og_3d = ol.create_3d_array_from_tiff(img_data_path)

    # stitched array (Stardist)
    sd_path = r'C:\Segmentation_working_area\stardist_testdata\masks'
    sd_3d = ol.create_3d_array(sd_path)
    sd_stitch, sd_nlen = ol.calc_all_overlaps(sd_3d)
    sd_bright = ol.calc_brightness(og_3d, sd_stitch, sd_nlen)

    # choose example neuron with z>20;
    # N > 16: [1, 13, 20, 23, 26, 29, 37, 39, 45, 47, 49, 52, 67]
    example_dist = sd_bright['1']
    # reshape distribution
    example_dist = example_dist.reshape(1, -1)

    gmm = GMM(n_components=2)

    model = gmm.fit(example_dist)
    x = np.linspace(1, len(example_dist), len(example_dist))


    print('Done with GMM')
    return


def what_is_x_when_y_is(input, x, y):
    order = y.argsort()
    x = x[order]
    y = y[order]

    # finds closest x-index of y-value coming from the left side
    return x[y.searchsorted(input, 'left')]


def brightness_histograms(brightness_dict: dict):
    plt.figure()
    # make a histogram for every entry in the brightness dict
    for k, v in brightness_dict.items():
        # x_means = []
        # try:
        #     x_means, g1, g2 = ol.calc_means_via_brightnesses(v, 0)
        # except RuntimeError:
        #     print(f'could not fit neuron: {k}')

        x_lin = np.arange(1, len(v) + 1)

        plt.plot(x_lin, v, color='b', label='Data')

        # if x_means:
        #     plt.plot(x_lin, g1, color='g', label='Fit1')
        #     plt.plot(x_lin, g2, color='r', label='Fit2')

        plt.title('Neuron: ' + str(k) + ' brightness')
        plt.ylabel('avg. brightness')
        plt.xlabel('Slice (relative)')

        sv_nm = r'C:\Segmentation_working_area\brightnesses'
        sv_nm = os.path.join(sv_nm, 'gt_brightness_after_stitching_' + str(k) + '.png')
        plt.tight_layout()
        plt.savefig(sv_nm)

    return


def bipartite_stitching(array_3d, verbose=0):

    # iterate over slices and their neurons and calculate the best matches using bipartite matching

    print(f'Starting with stitching. Array shape: {array_3d.shape}')
    num_slices = len(array_3d)

    # Initialize output matrix: full_3d_mask
    # Dimensions: ZXY
    output_3d_mask = np.zeros_like(array_3d)

    global_current_neuron = 1

    # loop over slices
    for i_slice in range(num_slices-1):
        print(f'--- Slice: {i_slice}')

        this_slice = array_3d[i_slice]
        next_slice = array_3d[i_slice + 1]

        this_slice_candidates = list()
        this_slice_candidates = create_matches_list(this_slice, next_slice)

        # Bipartite matching after creating overlap list for all neurons on slice
        bp_matches = list()
        bp_matches = calc_bipartite_matches(this_slice_candidates, 2)

        # rename neurons on existing slice according to best match
        for match in bp_matches:
            next_slice = np.where(next_slice == match[1], match[0], next_slice)

        array_3d[i_slice + 1] = next_slice

    # renaming all found neurons in array; in a sorted manner
    sorted_stitched_array = renaming_stitched_array(array_3d)

    return sorted_stitched_array    # stitched_array


def create_matches_list(slice_1, slice_2):

    # find all matches of a given neuron in the next slice
    neurons_this_slice = np.unique(slice_1)
    bip_list = list()

    # iterate over all neurons found in array[i_slice]
    for this_neuron in neurons_this_slice:
        bip_inter = list()
        if this_neuron == 0:
            continue

        print(f'... Neuron: {int(this_neuron)}')
        # new unique name for this_neuron
        # unique_neuron_this_slice = get_node_name(i_slice, this_neuron)

        # Get the initial mask, which will be propagated across slices
        this_mask_binary = (slice_1 == this_neuron)

        # overlap
        this_overlap_neurons = np.unique(this_mask_binary * slice_2)

        for overlap_neuron in this_overlap_neurons:
            if overlap_neuron == 0:
                continue
            overlap_slice = slice_2 == overlap_neuron
            overlap_slice = overlap_slice * this_mask_binary
            overlap_area = np.count_nonzero(overlap_slice)

            bip_inter.append([int(this_neuron), int(overlap_neuron), int(overlap_area)])

        bip_list.extend(bip_inter)

    # return a list of lists with all matches and their overlaps
    return bip_list


def renaming_stitched_array(arr):
    """
    Takes an array and changes the values of masks, so that it starts at 1 on slice 1 and increases consistently
    Parameters
    ----------
    arr : numpy array (3d)

    Returns
    -------
    sorted array
    """
    print(f'Starting to rename stitched array')
    arr = np.where(arr > 0, arr + 10000, arr)
    uniq_arr = np.unique(arr)
    uniq_arr = np.delete(uniq_arr, np.where(uniq_arr == 0))

    new_ids = list(range(1, len(uniq_arr) + 1))
    mapped_dict = dict(zip(uniq_arr, new_ids))

    for k, v in mapped_dict.items():
        arr = np.where((arr == k), v, arr)

    return arr

sd_dir = r'C:\Segmentation_working_area\data\stardist_raw\3d'
sd_files = [os.path.join(sd_dir, f.name) for f in os.scandir(sd_dir) if f.is_file()]
sd_array = np.load(sd_files[4])

print('Start with bipartite shit')
rm_array = ol.remove_large_areas(sd_array, 1000)
bp_array = ol.bipartite_stitching(sd_array)

print('Done')