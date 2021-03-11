import segmentation.util.overlap as ol
import segmentation.util.stardist_seg as sd
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import perform_preprocessing
from segmentation.util.prealignment_test import rigid_prealignment_pipeline

import numpy as np
import os
import tifffile as tiff


def segment_full_video(video_path,
                       preprocessing=PreprocessingSettings(do_filtering=False,alpha=1.0),
                       num_slices=33,
                       options={},
                       start_volume,
                       num_frames):

    # Some vars, that need to be in options object
    length_upper_cutoff = 12
    length_lower_cutoff = 3

    # we won't read in the WHOLE video!
    # add Option: range of timepoints/volumes, that shall be analyzed
    option_volumes = [100, 500]

    for i in range(start_volume, start_volume + num_frames):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        # preprocess
        volume = perform_preprocessing(volume)
        
        # segment the volume using Stardist
        masks = sd.segment_with_stardist_pipeline(volume)

        # process masks: remove large areas, stitch, split long neurons, remove short neurons
        masks = ol.remove_large_areas(masks)
        stitched_masks, df_with_centroids = ol.bipartite_stitching(masks)
        neuron_lengths = ol.get_neuron_lengths_dict(stitched_masks)

        brightnesses, brightnesses_global_z = ol.calc_brightness(volume, stitched_masks, neuron_lengths)
        split_masks, split_lengths, split_brightnesses, current_global_neuron = ol.split_long_neurons(stitched_masks,
                                                                                                      neuron_lengths,
                                                                                                      brightnesses,
                                                                                                      len(neuron_lengths),
                                                                                                      length_upper_cutoff)
        final_masks, final_neuron_lengths, removed_neurons_list = ol.remove_short_neurons(split_masks,
                                                                                    split_lengths,
                                                                                    length_lower_cutoff)

        # save tiff of all masks


        # metadata dictionary
        # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}

        # ad centroids: skimage.measure

    return final_masks, metadata_dictionary
