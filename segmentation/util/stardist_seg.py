from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import tifffile as tiff
from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D


def segment_with_stardist(vol_path):
    # segments a volume (via path)
    # vars set
    # Stardist models: versatile showed best results

    # TODO keep models available
    model_list = ['2D_paper_dsb2018', '2D_versatile_fluo', '2D_demo']
    mdl = model_list[1]  # '2D_versatile_fluo'
    axis_norm = (0, 1)
    n_channel = 1

    # logging/redirecting stdout
    log_file = "./log_" + mdl + ".log"

    with open(log_file, 'w') as log:
        # load stardist model
        # model = StarDist2D.from_pretrained(mdl)
        lukas_model = StarDist2D(None, name='stardistNiklas', basedir='C:\Segmentation_working_area')

        # open volume.tif
        with tiff.TiffFile(vol_path) as vol:
            # initialize output dimensions
            z = len(vol.pages)
            xy = vol.pages[1].shape
            output_file = np.zeros((z, *xy))    # '*' = tuple unpacking

            # iterate over images to run stardist on single images
            for idx, page in enumerate(vol.pages):
                img = page.asarray()
                print(idx, img.shape, file=log)

                # normalizing images (stardist function)
                img = normalize(img, 1, 99.8, axis=axis_norm)

                # run the prediction
                # labels, details = model.predict_instances(img)
                labels, details = lukas_model.predict_instances(img)

                # save labels in 3D array for output
                output_file[idx] = labels

    return output_file


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
