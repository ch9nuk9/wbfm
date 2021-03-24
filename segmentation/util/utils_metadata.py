"""
The metadata generator for the segmentation pipeline
"""
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


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
