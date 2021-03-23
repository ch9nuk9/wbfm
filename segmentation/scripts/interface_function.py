"""
Maybe
"""
import segmentation.util.overlap as ol
import segmentation.util.stardist_seg as sd
from segmentation.util.pipeline_helpers import get_metadata_dictionary
from segmentation.util.pipeline_helpers import get_stardist_model
from segmentation.util.pipeline_helpers import perform_post_processing_2d
from segmentation.util.pipeline_helpers import perform_post_processing_3d
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import perform_preprocessing
from segmentation.util.prealignment_test import rigid_prealignment_pipeline
from tqdm import tqdm
import pickle
import numpy as np
import os
import tifffile as tiff
import argh


def segment_full_video(video_path,
                       start_volume,
                       num_frames,
                       stardist_model_name='lukas',
                       preprocessing=PreprocessingSettings(do_filtering=False, do_rigid_alignment=True,
                                                           alpha=1.0, final_dtype='uint16'),
                       num_slices=33,
                       to_remove_border=True,
                       options={}):
    """
    Function to segment, stitch and curate a recording on a volume-to-volume basis.

    Parameters
    ----------
    video_path : str
        path to recording
    start_volume : int
        starting frame in absolute number starting from 0
    num_frames : int
        number of volumes to be analyzed
    stardist_model_name : str
        Name/alias of the StarDist model to be used for segmentation. I.e. 'lukas', 'charlie', 'versatile'
        see also: get_stardist_model()
    preprocessing : PreprocessingSettings() object
        Object containing preprocessing options
    num_slices : int
        number of total Z-slices per volume
    to_remove_border : boolean
        Flag for zeroing edge areas, because of edge effect artefacts in segmentations of prealigned volumes
    flag_3d : Boolean
        Flag for using Stardists' 3D segmentation
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
    output_folder = None
    border_width_to_be_removed = 100
    # maybe the preprocessing options too

    # initialization of variables
    output_folder = os.path.split(video_path)[0]
    output_fname = os.path.join(output_folder, 'segmented_masks.btf')
    metadata = dict()

    # get stardist model object
    sd_model = get_stardist_model(stardist_model_name)

    for i in tqdm(list(range(start_volume, start_volume + num_frames))):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        # preprocess
        volume = perform_preprocessing(volume, preprocessing)

        # segment the volume using Stardist
        print('--- Segmentation ---')
        segmented_masks = sd.segment_with_stardist_2d(volume, sd_model)

        # process masks: remove large areas, stitch, split long neurons, remove short neurons
        print('---- Post-processing ----')
        final_masks = perform_post_processing_2d(segmented_masks,
                                                 volume,
                                                 border_width_to_be_removed,
                                                 to_remove_border,
                                                 length_upper_cutoff,
                                                 length_lower_cutoff)

        print('----- Saving TIF -----')
        # concatenate masks to tiff file and save
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

    # saving metadata
    metadata_filename = os.path.join(output_folder, 'metadata.pickle')
    with open(metadata_filename, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)

    print(f'Done with segmentation pipeline! Data saved at {output_folder}')

    return output_fname, metadata


def segment_full_video_3d(video_path,
                          start_volume,
                          num_frames,
                          stardist_model_name='lukas',
                          preprocessing=PreprocessingSettings(do_filtering=False, do_rigid_alignment=True, alpha=1.0),
                          num_slices=33,
                          to_remove_border=True,
                          do_post_processing=False,
                          options={}):
    """
    Function to segment (& stitch and curate) a recording on a volume-to-volume basis.
    Segments a 3D volume with a self-trained network.

    Parameters
    ----------
    video_path : str
        path to recording
    start_volume : int
        starting frame in absolute number starting from 0
    num_frames : int
        number of volumes to be analyzed
    stardist_model_name : str
        Name/alias of the StarDist model to be used for segmentation. I.e. 'lukas', 'charlie', 'versatile'
        see also: get_stardist_model()
    preprocessing : PreprocessingSettings() object
        Object containing preprocessing options
    num_slices : int
        number of total Z-slices per volume
    to_remove_border : boolean
        Flag for zeroing edge areas, because of edge effect artefacts in segmentations of prealigned volumes
    flag_3d : Boolean
        Flag for using Stardists' 3D segmentation
    do_post_processing : boolean
        Flag for post-processing, i.e. splitting long neurons, removing short neurons.
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
    output_folder = None
    border_width_to_be_removed = 100

    # initialization of variables
    output_folder = os.path.split(video_path)[0]
    output_fname = os.path.join(output_folder, 'segmented_masks.btf')
    metadata = dict()

    # get stardist model object
    sd_model = get_stardist_model(stardist_model_name)

    for i in tqdm(list(range(start_volume, start_volume + num_frames))):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        # preprocess
        volume = perform_preprocessing(volume, preprocessing)

        # segment the volume using Stardist
        masks = sd.segment_with_stardist_3d(volume, sd_model)

        if do_post_processing:
            print('--- Post processing Start ---')
            # process masks: remove large areas, stitch, split long neurons, remove short neurons
            masks, neuron_lengths, brightness = perform_post_processing_3d(masks,
                                                                           volume,
                                                                           border_width_to_be_removed,
                                                                           to_remove_border,
                                                                           length_upper_cutoff,
                                                                           length_lower_cutoff)

        # concatenate masks to tiff file and save
        if i == start_volume:
            tiff.imwrite(output_fname,
                         masks,
                         append=False,
                         bigtiff=True)
        else:
            tiff.imwrite(output_fname,
                         masks,
                         append=True,
                         bigtiff=True)

        # metadata dictionary
        # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
        meta_df = get_metadata_dictionary(masks, volume)
        metadata[i] = meta_df

    # saving metadata
    metadata_filename = os.path.join(output_folder, 'metadata.pickle')
    with open(metadata_filename, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)

    print(f'Done with segmentation pipeline! Data saved at {output_folder}')

    return output_fname, metadata


# dispatching the argh parser, so that the functions can be run on CLI
parser = argh.ArghParser()
parser.add_commands([segment_full_video, segment_full_video_3d])

if __name__ == '__main__':
    parser.dispatch()
