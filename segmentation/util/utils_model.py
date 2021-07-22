"""
StarDist functions for segmentation
"""
import numpy as np
import skimage
import stardist.models
from stardist.models import Config3D, StarDistData3D, StarDist3D, StarDist2D
import os
from csbdeep.utils import Path, normalize


def get_stardist_model(model_name: str, folder: str = None, verbose: int = 0) -> stardist.models.StarDist3D:
    """
    Fetches the wanted StarDist model for segmenting images.
    Add new StarDist models as an alias below (incl. sd_options)

    Parameters
    ----------
    model_name : str
        Name of the wanted model. See
    folder : str
        Path in which the stardist models are saved
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    model : StarDist model
        model object of the StarDist model. Can be directly used for segmenting

    """

    if verbose >= 1:
        print(f'Getting Stardist model: {model_name}')
    # all self-trained StarDist models reside in that folder. 'nt' for windows, when working locally
    if folder is None:
        if os.name == 'nt':
            folder = Path(r'Y:\shared_projects\wbfm\TrainedStardist')
        else:
            folder = Path('/groups/zimmer/shared_projects/wbfm/TrainedStardist')

    # available models' aliases
    sd_options = ['versatile', 'lukas', 'charlie', 'charlie_3d', 'charlie_3d_party']

    # create aliases for each model_name
    if model_name == 'versatile':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    elif model_name == 'lukas':
        model = StarDist2D(None, name='stardistNiklas', basedir=folder)
    elif model_name == 'charlie':
        model = StarDist2D(None, name='stardistCharlie', basedir=folder)
    elif model_name == 'charlie_3d':
        model = StarDist3D(None, name='Charlie100-3d', basedir=folder)
    elif model_name == 'charlie_3d_party':
        model = StarDist3D(None, name='Charlie100-3d-party', basedir=folder)
    else:
        raise NameError(f'No StarDist model found using {model_name}! Current models are {sd_options}')

    return model


def segment_with_stardist_2d(vol, model=None, zero_out_borders=False, verbose=0) -> np.array:
    """
    Segments slices of a 3D numpy array (input) and outputs their masks.
    Best model (so far) is Lukas' self-trained 2D model
    Parameters
    ----------
    vol : 3D numpy array
        Original image array
    model : StarDist2D model object
        Object of a Stardist model, which will be used for prediction
    zero_out_borders : bool
        Whether to calculate the object borders (which may be touching) and zero them out (to prevent touching)
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    segmented_masks : 3D numpy array
        2D segmentations of slices concatenated to a 3D array. Each slice has unique values within
        a slice, but will be duplicated across slices (needs to be stitched in next step)!
    """

    if model is None:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')

    if verbose >= 1:
        print(f'Start of 2D segmentation.')

    # initialize output dimensions and other variables
    z = len(vol)
    # xy = vol.shape[1:]
    segmented_masks = np.zeros_like(vol)    # '*' = tuple unpacking
    if zero_out_borders:
        boundary = np.zeros_like(segmented_masks, dtype='bool')
    # segmented_masks = np.zeros((z, *xy))    # '*' = tuple unpacking
    axis_norm = (0, 1)
    # n_channel = 1

    # iterate over images to run stardist on single images
    for idx, plane in enumerate(vol):
        img = plane

        # normalizing images (stardist function)
        img = normalize(img, 1, 99.8, axis=axis_norm)

        # run the prediction
        labels, details = model.predict_instances(img)

        # save labels in 3D array for output
        segmented_masks[idx] = labels

        if verbose >= 2:
            print(f"Found {len(np.unique(labels))} neurons on slice {idx}/{z}")

        if zero_out_borders:
            # Postprocess to add separation between labels
            # From: watershed.py in 3DeeCellTracker
            labels_bd = skimage.segmentation.find_boundaries(labels, connectivity=2, mode='outer', background=0)

            boundary[idx, :, :] = labels_bd

            # save labels in 3D array for output
            segmented_masks[idx] = labels

    if zero_out_borders:
        segmented_masks[boundary == 1] = 0

    return segmented_masks


def segment_with_stardist_3d(vol, model=None, verbose=0) -> np.array:
    """
    Segments a 3D volume using stardists 3D-segmentation.
    For now, only one self-trained 3D model is available.

    Parameters
    ----------
    vol : 3D numpy array
        3D array of volume to segment
    model : StarDist3D object
        StarDist3D model to be used for segmentation; default = Charlies first trained 3D model
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    labels : 3D numpy array
        3D array with segmented masks. Each mask should have a unique ID/value.
    """

    if verbose >= 1:
        print('Start of 3D segmentation')

    # initialize variables
    axis_norm = (0, 1, 2)
    n_channel = 1

    # normalizing images (stardist function)
    img = normalize(vol, 1, 99.8, axis=axis_norm)

    # run the prediction
    labels, details = model.predict_instances(img)

    return labels
