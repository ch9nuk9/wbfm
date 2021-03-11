import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import tqdm
from stardist.models import StarDist2D


def get_metadata_dictionary(masks, original_vol):
    """
    Creates a dataframe with metadata ('total_brightness', 'neuron_volume', 'centroids') for a volume
    Parameters
    ----------
    masks : 3D numpy arrray
        contains the segmented and stitched masks
    original_vol : 3d numpy arrray
        original volume of the recording, which was segmented into 'masks'.
        Contains the actual brightness values.

    Returns
    -------
    df : pandas dataframe
        for each neuron (=row) the total brightness, volume and centroid is saved
        dataframe(columns = 'total_brightness', 'neuron_volume', 'centroids';
                  rows = neuron #)
    """
    # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
    neurons = np.unique(masks)
    neurons = np.delete(neurons, np.where(neurons == 0))
    # print(neurons)

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
    # cols = (total brightness, volume, centroids)
    # rows = neuron ID
    df = pd.DataFrame(list(zip(brightnesses, neuron_volumes, all_centroids)),
                      index=neurons,
                      columns=['total_brightness', 'neuron_volume', 'centroids'])

    return df


def get_stardist_model(model_name):
    """
    Fetches the wanted StarDist model for segmenting images.
    Add new StarDist models as an alias below (incl. sd_options)

    Parameters
    ----------
    model_name : str
        Name of the wanted model. See

    Returns
    -------
    model : StarDist model
        model object of the StarDist model. Can be directly used for segmenting

    """
    # all self-trained StarDist models reside in that folder
    folder = r'/groups/zimmer/shared_projects/wbfm/TrainedStardist'
    sd_options = ['versatile', 'lukas', 'charlie']

    # create aliases for each model_name
    if model_name == 'versatile':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    elif model_name == 'lukas':
        model = StarDist2D(None, name='stardistNiklas', basedir=folder)
    elif model_name == 'charlie':
        model = StarDist2D(None, name='stardistCharlie', basedir=folder)
    else:
        print(f'No StarDist model found using {model_name}! Current models are {sd_options}')

    return model
