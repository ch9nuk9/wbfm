import concurrent
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from skimage import measure
from tqdm.auto import tqdm

from wbfm.utils.external.utils_neuron_names import int2name_using_mode


def region_props_all_volumes(reindexed_masks, red_video, green_video,
                             frame_list,
                             params_start_volume,
                             name_mode) -> Tuple[dict, dict]:
    """

    Parameters
    ----------
    name_mode
    reindexed_masks
    red_video
    green_video
    frame_list
    params_start_volume

    Returns
    -------
    Two nested dictionaries: red_all_neurons, green_all_neurons
        Outer keys are volume indices, and inner keys are neuron id + property name (tuple)
    """
    red_all_neurons = {}
    green_all_neurons = {}

    def _parallel_func(i_volume):
        i_mask = i_volume - params_start_volume
        this_mask_volume = reindexed_masks[i_mask, ...]
        this_green_volume = green_video[i_volume, ...]
        this_red_volume = red_video[i_volume, ...]
        # mask2final_name = mask2final_name_per_volume[i_volume]
        red_one_vol, green_one_vol = region_props_one_volume(
            this_mask_volume,
            this_red_volume,
            this_green_volume,
            name_mode
        )
        red_all_neurons[i_volume] = red_one_vol
        green_all_neurons[i_volume] = green_one_vol

    with tqdm(total=len(frame_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_parallel_func, i): i for i in frame_list}
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                pbar.update(1)

    return red_all_neurons, green_all_neurons


def region_props_one_volume(this_mask_volume,
                            this_red_volume,
                            this_green_volume,
                            name_mode):
    """

    Parameters
    ----------
    name_mode
    this_green_volume
    this_mask_volume
    this_red_volume

    Returns
    -------
    Two dictionaries with keys that are 2-tuples (i_final_neuron, property_name) with properties:
        props_to_save = ['area', 'weighted_centroid', 'image_intensity', 'label']

    """
    props_to_save = ['area', 'weighted_centroid', 'intensity_image', 'label']
    opt = dict(name_mode=name_mode, props_to_save=props_to_save)
    red_neurons_one_volume = regionprops_one_volume(this_mask_volume, this_red_volume, **opt)
    green_neurons_one_volume = regionprops_one_volume(this_mask_volume, this_green_volume, **opt)

    return red_neurons_one_volume, green_neurons_one_volume


def regionprops_one_volume(mask, data, props_to_save, name_mode):
    neurons_one_volume = {}
    props = measure.regionprops(mask, intensity_image=data)

    for this_neuron in tqdm(props, leave=False):
        seg_index = this_neuron['label']
        # final_index = mask2final_name[seg_index]
        # NOTE: This must be synchronized to the df_global_tracks naming and indexing scheme!
        key_base = (int2name_using_mode(seg_index, name_mode),)

        for this_prop in props_to_save:
            key = key_base + (this_prop,)
            if this_prop == 'intensity_image':
                neurons_one_volume[key] = np.sum(this_neuron[this_prop])
            else:
                neurons_one_volume[key] = this_neuron[this_prop]

    return neurons_one_volume


def _convert_nested_dict_to_dataframe(coords: list, frame_list: list, nested_dict_all_neurons: dict):
    # Convert nested dict of volumes to final dataframes
    sz_one_neuron = len(frame_list)
    i_start = min(list(nested_dict_all_neurons.keys()))
    tmp_neurons = defaultdict(lambda: np.zeros(sz_one_neuron))
    volume_indices = list(nested_dict_all_neurons.keys())
    volume_indices.sort()
    for i_vol in volume_indices:
        props = nested_dict_all_neurons[i_vol]
        i_vol_numpy = i_vol - i_start
        for key in props.keys():
            if 'weighted_centroid' in key:
                # Later formatting expects this to be split, i.e. z x y are separate columns
                for i, c in enumerate(coords):
                    new_key = (key[0], c)
                    tmp_neurons[new_key][i_vol_numpy] = props[key][i]
            else:
                tmp_neurons[key][i_vol_numpy] = props[key]
    df_neurons = pd.DataFrame(tmp_neurons, index=volume_indices)
    return df_neurons
