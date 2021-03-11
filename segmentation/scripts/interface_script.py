import segmentation.util.overlap as ol
import segmentation.util.stardist_seg as sd
from segmentation.util.pipeline_helpers import get_metadata_dictionary
from segmentation.util.pipeline_helpers import get_stardist_model
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import perform_preprocessing
from segmentation.util.prealignment_test import rigid_prealignment_pipeline
from tqdm import tqdm

import numpy as np
import os
import tifffile as tiff
from collections import defaultdict


def segment_full_video(video_path,
                       start_volume,
                       num_frames,
                       stardist_model_name='versatile',
                       preprocessing=PreprocessingSettings(do_filtering=False,alpha=1.0),
                       num_slices=33,
                       to_remove_border=True,
                       options={}):
    """

    Parameters
    ----------
    video_path : str
        path to recording
    start_volume : int
        starting frame in absolute number starting from 0
    num_frames : int
        number of volumes to be analyzed
    stardist_model_name : str
        Name/alias of the StarDist model to be used for segmentation
    preprocessing : PreprocessingSettings() object
        Object containing preprocessing options
    num_slices : int
        number of total Z-slices per volume
    options : object
        contains further necessary options for running

    Returns
    -------
    output_fname : str
        path to the TIF-file of segmented masks
    metadata : dict
        dictionary containing metadata per volume and neuron. Each volume's dataframe has N rows, where
        N = number of unique neurons in that slice and 3 columns = ('total_brightness', 'neuron_volume', 'centroids')
        {volume_number : [dataframe('total_brightness', 'neuron_volume', 'centroids')]}

        Use the column name to index directly into the dataframe to get all e.g. centroid values
    """

    # TODO add the following values to an options object!
    # Some vars, that need to be in options object
    length_upper_cutoff = 12
    length_lower_cutoff = 3

    # we won't read in the WHOLE video!
    # add Option: range of timepoints/volumes, that shall be analyzed
    option_volumes = [100, 500]

    # initialization of variables
    output_fname = os.path.join(os.path.split(video_path)[0], 'masks.tif')
    metadata = defaultdict(dict)

    # get stardist model object
    sd_model = get_stardist_model(stardist_model_name)

    for i in tqdm(list(range(start_volume, start_volume + num_frames))):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        # preallocate single volume

        # preprocess
        volume = perform_preprocessing(volume, preprocessing)

        # segment the volume using Stardist
        masks = sd.segment_with_stardist_pipeline(volume, sd_model)

        # process masks: remove large areas, stitch, split long neurons, remove short neurons
        masks = ol.remove_large_areas(masks)
        stitched_masks, df_with_centroids = ol.bipartite_stitching(masks)
        neuron_lengths = ol.get_neuron_lengths_dict(stitched_masks)

        brightnesses, brightnesses_global_z = ol.calc_brightness(volume, stitched_masks, neuron_lengths)
        split_masks, split_lengths,\
        split_brightnesses, current_global_neuron = ol.split_long_neurons(stitched_masks,
                                                                          neuron_lengths,
                                                                          brightnesses,
                                                                          len(neuron_lengths),
                                                                          length_upper_cutoff)

        final_masks, final_neuron_lengths, removed_neurons_list = ol.remove_short_neurons(split_masks,
                                                                                          split_lengths,
                                                                                          length_lower_cutoff)

        if to_remove_border:
            final_masks = remove_border(final_masks, border=100)

        # concatenate masks to tiff file
        if i == start_volume:
            tiff.imwrite(output_fname,
                         final_masks,
                         append=False,
                         bigtiff=True)
        else:
            tiff.imwrite(output_fname,
                         final_masks,
                         append=True,
                         bigtiff=True)

        # metadata dictionary
        # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
        meta_df = get_metadata_dictionary(final_masks, volume)
        metadata[i] = meta_df

    return output_fname, metadata



def remove_border(masks, border=100):
    _, x_sz, y_sz = masks.shape

    masks[:,:border,:] = 0.0
    masks[:,(x_sz-border):,:] = 0.0
    masks[:,:,:border] = 0.0
    masks[:,:,(y_sz-border):] = 0.0

    return masks
