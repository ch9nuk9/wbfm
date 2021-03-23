import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from stardist.models import Config3D, StarDistData3D, StarDist3D, StarDist2D
import os
from pathlib import Path
import segmentation.util.overlap as ol


def get_metadata_dictionary(masks, original_vol):
    """
    Creates a dataframe with metadata ('total_brightness', 'neuron_volume', 'centroids') for a volume
    Parameters
    ----------
    masks : 3D numpy array
        contains the segmented and stitched masks
    original_vol : 3d numpy array
        original volume of the recording, which was segmented into 'masks'.
        Contains the actual brightness values.
    planes = : dict(list)
        Dictionary containing a list of Z-slices on which a neuron is present
        neuron_ID = 1
        planes[neuron_ID] # [12, 13, 14]

    Returns
    -------
    df : pandas dataframe
        for each neuron (=row) the total brightness, volume and centroid is saved
        dataframe(columns = 'total_brightness', 'neuron_volume', 'centroids';
                  rows = neuron #)
    """
    # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
    neurons_list = np.unique(masks)
    neurons_list = np.delete(neurons_list, np.where(neurons_list == 0))
    neurons = []
    # create list of integers for each neuron ID (instead of float)
    for x in neurons_list:
        neurons.append(int(x))

    all_centroids = []
    neuron_volumes = []
    brightnesses = []

    for n in neurons:
        neuron_mask = masks == n
        neuron_vol = np.count_nonzero(neuron_mask)
        total_brightness = np.sum(original_vol[neuron_mask])

        neuron_label = label(neuron_mask)
        centroids = regionprops(neuron_label)[0].centroid

        brightnesses.append(total_brightness)
        neuron_volumes.append(neuron_vol)
        all_centroids.append(centroids)

    # create dataframe with
    # cols = (total brightness, volume, centroids, z_planes)
    # rows = neuron ID
    df = pd.DataFrame(list(zip(brightnesses, neuron_volumes, all_centroids)),
                      index=neurons,
                      columns=['total_brightness', 'neuron_volume', 'centroids'])

    return df


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
            folder = Path(r'C:\Segmentation_working_area\stardist_models')
        else:
            folder = Path(r'C:\Segmentation_working_area\stardist_models')

    # available models' aliases
    sd_options = ['versatile', 'lukas', 'charlie', 'charlie_3d']
    sd_names = ['2D_versatile_fluo', 'stardistNiklas', 'starDistCharlie', 'Charlie100-3d']
    # TODO create a dictionary with key = 'alias', value = name
    # but it makes no sense, if we have 3 exceptions for 4 models

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


def perform_post_processing_2d(mask_array, img_volume, remove_border_width, remove_border_flag=True,
                               upper_length_threshold=12, lower_length_threshold=3):
    """
    Performs some post-processing steps including: Splitting long neurons, removing short neurons and
    removing too large areas

    Parameters
    ----------
    mask_array : 3D numpy array
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    remove_border_width : int
        within that distance to border, artefacts/masks will be removed
    remove_border_flag : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed

    Returns
    -------
    final_masks : 3D numpy array
        3D array of masks after post-processing

    """

    masks = ol.remove_large_areas(mask_array)
    stitched_masks, df_with_centroids = ol.bipartite_stitching(masks)
    neuron_lengths = ol.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = ol.calc_brightness(img_volume, stitched_masks, neuron_lengths)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        ol.split_long_neurons(stitched_masks,
                              neuron_lengths,
                              brightnesses,
                              len(neuron_lengths),
                              upper_length_threshold,
                              neuron_planes)

    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        ol.remove_short_neurons(split_masks,
                                split_lengths,
                                lower_length_threshold,
                                split_brightnesses,
                                split_neuron_planes)

    if remove_border_flag is True:
        final_masks = remove_border(final_masks, remove_border_width)

    return final_masks


def perform_post_processing_3d(stitched_masks, img_volume, remove_border_width, remove_border_flag=True,
                               upper_length_threshold=12, lower_length_threshold=3):
    """
    Performs post-processing of segmented masks. Includes: splitting long masks, removing large areas,
    removing short masks as well as removing artefacts close to the border

    Parameters
    ----------
    stitched_masks
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    remove_border_width : int
        within that distance to border, artefacts/masks will be removed
    remove_border_flag : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed

    Returns
    -------
    labels : 3D numpy array
        3D array of processed masks
    """

    stitched_masks = ol.remove_large_areas(stitched_masks)
    neuron_lengths = ol.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = ol.calc_brightness(img_volume, stitched_masks, neuron_lengths)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        ol.split_long_neurons(stitched_masks,
                              neuron_lengths,
                              brightnesses,
                              len(neuron_lengths),
                              upper_length_threshold,
                              neuron_planes)

    # remove short neurons
    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        ol.remove_short_neurons(split_masks,
                                split_lengths,
                                lower_length_threshold,
                                split_brightnesses,
                                split_neuron_planes)

    if remove_border_flag is True:
        final_masks = remove_border(final_masks, remove_border_width)

    return final_masks


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

    masks[:, :border, :] = 0.0
    masks[:, (x_sz - border):, :] = 0.0
    masks[:, :, :border] = 0.0
    masks[:, :, (y_sz - border):] = 0.0

    return masks
