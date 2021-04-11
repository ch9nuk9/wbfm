"""
The top level functions for segmenting a full (WBFM) recording.
"""

from tqdm import tqdm
import pickle
import os
import tifffile as tiff
# preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.preprocessing.utils_tif import perform_preprocessing
# postproc
from segmentation.util.utils_pipeline import perform_post_processing_2d
# metadata
from segmentation.util.utils_metadata import get_metadata_dictionary
from segmentation.util.utils_paths import get_output_fnames
# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Initialize sacred experiment
ex = Experiment()
ex.add_config(r'config\segment_config.yaml')
ex.observers.append(FileStorageObserver('runs'))


@ex.config
def cfg(video_path):
    # Check paths
    if video_path is None:
        print("Must path a valid video path!")
    assert os.path.exists(video_path)


@ex.automain
def segment2d(_config, _run):
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
        Name/alias of the StarDist model to be used for segmentation. I.e. 'lukas', 'charlie', 'versatile', 'Charlie100-3d'
        see also: get_stardist_model()
    preprocessing : yaml file for preprocessing settings
        see also PreprocessingSettings
    num_slices : int
        number of total Z-slices per volume
    to_remove_border : boolean
        Flag for zeroing edge areas, because of edge effect artefacts in segmentations of prealigned volumes
    flag_3d : Boolean
        Flag for using Stardists' 3D segmentation
    options : object
        contains further necessary options for running
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1
    DEBUG : bool
        flag for a 10 second run; segments one volume and writes small versions of all files

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
    sacred.commands.print_config(_run)

    # Initializing variables
    start_volume, num_frames, num_slices = _config['dataset_params'].values()
    video_path = _config['video_path']
    mask_fname, metadata_fname = get_output_fnames(video_path, _config)
    verbose = _config['verbose']
    metadata = dict()
    preprocessing_settings = PreprocessingSettings.load_from_yaml(
        _config['preprocessing_config']
    )
    if verbose >= 1:
        print("Loaded preprocessing_settings:")
        print(preprocessing_settings)

    # get stardist model object
    from segmentation.util.utils_model import get_stardist_model
    from segmentation.util.utils_model import segment_with_stardist_2d
    stardist_model_name = _config['segmentation_params']['stardist_model_name']
    sd_model = get_stardist_model(stardist_model_name, verbose=verbose-1)

    if verbose >= 1:
        print('--- Starting loop through volumes ---')
    for i in tqdm(list(range(start_volume, start_volume + num_frames))):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        if _config['DEBUG']:
            break

        # preprocess
        volume = perform_preprocessing(volume, preprocessing_settings)

        # segment the volume using Stardist
        if verbose >= 2:
            print('--- Segmentation ---')
        segmented_masks = segment_with_stardist_2d(volume, sd_model, verbose=verbose-1)

        # process masks: remove large areas, stitch, split long neurons, remove short neurons
        if verbose >= 2:
            print('---- Post-processing ----')
        opt = _config['postprocessing_params']
        final_masks = perform_post_processing_2d(segmented_masks,
                                                 volume,
                                                 **opt,
                                                 verbose=verbose-1)

        if verbose >= 2:
            print('----- Saving to BIG-TIF -----')

        # concatenate masks to tiff file and save
        if i == start_volume:
            tiff.imwrite(mask_fname,
                         final_masks,
                         append=False,
                         bigtiff=True)
        else:
            tiff.imwrite(mask_fname,
                         final_masks,
                         append=True,
                         bigtiff=True)

        # metadata dictionary
        # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
        meta_df = get_metadata_dictionary(final_masks, volume)
        metadata[i] = meta_df

    # saving metadata
    with open(metadata_fname, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)

    if verbose >= 1:
        print(f'Done with segmentation pipeline! Data saved at {mask_fname}')
