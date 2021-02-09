"""

Main function of segmenting volumes. It generates predicted masks of volumes, which have
been given as input.

"""
import segmentation.util.overlap as ol

def main_segmentation_sd(volume_path, align_flag=False):
    """
    Top level function; Pipeline from raw data volume (input) to stitched segmented 3D masks (output)

    Parameters
    ----------
    volume_path : str
        Absolute volume path
    align_flag : bool
        Flag for optional pre-alignment

    Returns
    -------
    stitched_array : npy array
        3D array of segmented masks, stitched together with unique IDs
    neuron_lengths : dict
        Dictionary of "tube" lengths of identified and segmented neurons
    brightnesses : dict
        Dictionary of average brightnesses per neuron per slice

    """
    print('Starting to segment and stitch')
    # Read in volume and pass it as array to calculation function

    raw_array, algo_array = ol.array_dispatcher(volume_path, align_flag)
    print(f'.. done with data preparation --> stitching')

    # calculations and stitching. Optionally, set max/min neuron length here
    stitched_array, neuron_lengths, brightnesses = ol.level2_overlap(raw_array, algo_array)

    # possible saving function and neuron lengths histogram plotting

    # # first, create results folder if not existent
    # results_path = os.path.join(os.path.split(algo_data_path)[0], 'results')
    # if not os.path.exists(results_path):
    #     os.mkdir(results_path)
    # np.save(os.path.join(results_path, 'final_mask'), masks, allow_pickle=True)
    #
    # with open(os.path.join(results_path, 'lengths_and_brightnesses.pickle'), 'wb') as pickle_out:
    #     pickle.dump([neuron_lengths, brightnesses], pickle_out)
    #
    # # save length histograms
    # neuron_length_hist(neuron_lengths, results_path, 0)

    print(f'----- Done with all functions! -----')

    return stitched_array, neuron_lengths, brightnesses

# TODO decide where to cut off fliyback!