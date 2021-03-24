"""
StarDist functions for segmentation
"""
import numpy as np
from stardist.models import Config3D, StarDistData3D, StarDist3D, StarDist2D
import os
from csbdeep.utils import Path, normalize


def get_stardist_model(model_name, folder=None):
    """
    Fetches the wanted StarDist model for segmenting images.
    Add new StarDist models as an alias below (incl. sd_options)

    Parameters
    ----------
    model_name : str
        Name of the wanted model. See
    folder : str
        Path in which the stardist models are saved

    Returns
    -------
    model : StarDist model
        model object of the StarDist model. Can be directly used for segmenting

    """
    # all self-trained StarDist models reside in that folder. 'nt' for windows, when working locally
    if folder is None:
        if os.name == 'nt':
            folder = Path(r'/groups/zimmer/shared_projects/wbfm/TrainedStardist')
        else:
            folder = Path(r'/groups/zimmer/shared_projects/wbfm/TrainedStardist')

    # available models' aliases
    sd_options = ['versatile', 'lukas', 'charlie', 'charlie_3d']
    sd_names = ['2D_versatile_fluo', 'stardistNiklas', 'starDistCharlie', 'Charlie100-3d']

    # create aliases for each model_name
    if model_name == 'versatile':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    elif model_name == 'lukas':
        model = StarDist2D(None, name='stardistNiklas', basedir=folder)
    elif model_name == 'charlie':
        model = StarDist2D(None, name='stardistCharlie', basedir=folder)
    elif model_name == 'charlie_3d':
        model = StarDist3D(None, name='Charlie100-3d', basedir=folder)
    else:
        print(f'No StarDist model found using {model_name}! Current models are {sd_options}')

    return model


def segment_with_stardist_2d(vol, model=StarDist2D.from_pretrained('2D_versatile_fluo')):
    """
    Segments slices of a 3D numpy array (input) and outputs their masks.
    Best model (so far) is Lukas' self-trained 2D model
    Parameters
    ----------
    vol : 3D numpy array
        Original image array
    model : StarDist2D model object
        Object of a Stardist model, which will be used for prediction
    Returns
    -------
    segmented_masks : 3D numpy array
        2D segmentations of slices concatenated to a 3D array. Each slice has unique values within
        a slice, but will be duplicated across slices (needs to be stitched in next step)!
    """

    # initialize output dimensions and other variables
    z = len(vol)
    xy = vol.shape[1:]
    segmented_masks = np.zeros((z, *xy))    # '*' = tuple unpacking
    axis_norm = (0, 1)
    n_channel = 1

    # iterate over images to run stardist on single images
    for idx, plane in enumerate(vol):
        img = plane

        # normalizing images (stardist function)
        img = normalize(img, 1, 99.8, axis=axis_norm)

        # run the prediction
        labels, details = model.predict_instances(img)

        # save labels in 3D array for output
        segmented_masks[idx] = labels

    return segmented_masks


def segment_with_stardist_3d(vol, model):
    """
    Segments a 3D volume using stardists 3D-segmentation.
    For now, only one self-trained 3D model is available.

    Parameters
    ----------
    vol : 3D numpy array
        3D array of volume to segment
    model : StarDist3D object
        StarDist3D model to be used for segmentation

    Returns
    -------
    labels : 3D numpy array
        3D array with segmented masks. Each mask should have a unique ID/value.
    """

    # initialize variables
    axis_norm = (0, 1, 2)
    n_channel = 1

    # normalizing images (stardist function)
    img = normalize(vol, 1, 99.8, axis=axis_norm)

    # run the prediction
    labels, details = model.predict_instances(img)

    return labels
