from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from backports import cached_property
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None,
                                        z_to_xy_ratio=1, DEBUG=False):
    """
    Expects dataframe with positions, with column names either:
        legacy format: ['z_dlc', 'x_dlc', 'y_dlc']
        current format: ['z', 'x', 'y']

        And optionally: 'i_reindexed_segmentation' or 'label'
        (note: additional columns do not matter)

    Returns napari-ready format:
        A dict of options, with a nested dict 'properties' and a list 'data'
        'properties' has one entry, 'labels' = long list with all points at all time
        'dat' is a list of equal length with all the dimensions (tzxy)

    Parameters
    ----------
    z_to_xy_ratio
    df
    neuron_name_dict
    DEBUG

    Returns
    -------

    """
    df.replace(0, np.NaN, inplace=True)  # DLC uses all zeros as failed tracks

    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = get_names_from_df(df)
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = dict(label=[])
    for n in all_neurons:
        coords = ['z', 'x', 'y']
        zxy = np.array(df[n][coords])

        # if round_in_z:
        #     zxy[:, 0] = np.round(zxy[:, 0])
        zxy[:, 0] *= z_to_xy_ratio
        # zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        if n in neuron_name_dict:
            # label_vec[:] = this_name
            label_vec = [neuron_name_dict[n]] * len(df.index)
            if DEBUG:
                print(f"Found named neuron: {n} = {label_vec[0]}")
        else:
            # Get the index from the dataframe, or try to convert the column name into a label
            if 'i_reindexed_segmentation' in df[n]:
                label_vec = list(map(int, df[n]['i_reindexed_segmentation']))
            elif 'label' in df[n]:
                # For traces dataframe
                label_vec = [i for i in df[n]['label']]
            elif 'raw_neuron_ind_in_list' in df[n]:
                # For tracks dataframe
                label_vec = [i for i in df[n]['raw_neuron_ind_in_list']]
            else:
                label_vec = [name2int_neuron_and_tracklet(n) for _ in range(t_vec.shape[0])]

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)
    # Remove invalid positions
    # Some points are negative instead of nan
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]
    # Additionally remove invalid names
    to_keep = np.array([not np.isnan(p) for p in properties['label']])
    all_t_zxy = all_t_zxy[to_keep, :]
    properties['label'] = [cast_int_or_nan(p) for p, good in zip(properties['label'], to_keep) if good]

    # More info on text: https://github.com/napari/napari/blob/main/examples/add_points_with_text.py
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent',
               'text': {'text': 'label'},  # Can add color or size here
               'properties': properties, 'name': 'Neuron IDs', 'blending': 'additive',
               'visible': False}

    return options


@dataclass
class NapariPropertyHeatMapper:
    """Builds dictionaries to map segmentation labels to various neuron properties (e.g. average or max brightness)"""

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    @property
    def names(self):
        return get_names_from_df(self.red_traces)

    @property
    def vec_of_labels(self):
        return np.nanmean(self.df_labels.to_numpy(), axis=0).astype(int)

    @property
    def df_labels(self) -> pd.DataFrame:
        return self.red_traces.loc[:, (slice(None), 'label')]

    @property
    def mean_red(self):
        tmp1 = self.red_traces.loc[:, (slice(None), 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.red_traces.loc[:, (slice(None), 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    @property
    def mean_green(self):
        tmp1 = self.green_traces.loc[:, (slice(None), 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.green_traces.loc[:, (slice(None), 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    def count_nonnan(self) -> Dict[int, float]:
        num_nonnan = self.df_labels.count()
        val_to_plot = np.array(num_nonnan) / self.df_labels.shape[0]
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_red(self):
        val_to_plot = list(self.mean_red.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_red(self):
        val_to_plot = list(self.mean_red.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_green(self):
        val_to_plot = list(self.mean_green.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_green(self):
        val_to_plot = list(self.mean_green.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)


def property_vector_to_colormap(val_to_plot, vec_of_labels, cmap=plt.cm.plasma):
    prop = np.array(val_to_plot)
    prop_scaled = (
            (prop - prop.min()) / (prop.max() - prop.min())
    )  # matplotlib cmaps need values in [0, 1]
    colors = cmap(prop_scaled)
    prop_dict = dict(zip(vec_of_labels, colors))
    return prop_dict
