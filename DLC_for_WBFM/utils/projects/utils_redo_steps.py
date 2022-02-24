import logging

import numpy as np
import zarr
from DLC_for_WBFM.utils.external.utils_networkx import calc_bipartite_from_positions
from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from skimage.measure import regionprops
from tqdm.auto import tqdm


def remap_tracklets_to_new_segmentation(project_data: ProjectData, path_to_new_segmentation,
                                        DEBUG=False):

    old_seg = project_data.raw_segmentation
    new_seg = zarr.open(path_to_new_segmentation)
    red = project_data.red_data
    num_frames = project_data.num_frames
    old_df = project_data.df_all_tracklets
    try:
        logging.info("Converting dataframe to dense form, may take a while")
        if DEBUG:
            old_df = old_df[:5].sparse.to_dense()
        else:
            old_df = old_df.sparse.to_dense()
    except AttributeError:
        pass
    names = get_names_from_df(old_df)

    if DEBUG:
        num_frames = 5

    logging.info("Create mapping from old to new segmentation")
    all_old2new_idx = {}
    all_old2new_labels = {}
    for t in tqdm(range(num_frames)):
        this_img = red[t]

        new_centroids, new_labels = _get_props(new_seg[t], this_img)
        old_centroids, old_labels = _get_props(old_seg[t], this_img)

        new_c_array = np.array(list(new_centroids.values()))
        old_c_array = np.array(list(old_centroids.values()))

        old2new_idx, conf, _ = calc_bipartite_from_positions(old_c_array, new_c_array)
        old2new_labels = {old_labels[i1]: new_labels[i2] for i1, i2 in old2new_idx}

        all_old2new_idx[t] = dict(old2new_idx)
        all_old2new_labels[t] = old2new_labels

    logging.info("Updating tracklet dataframe using mapping")
    col_name1 = 'raw_segmentation_id'
    col_name2 = 'raw_neuron_ind_in_list'
    for n in tqdm(names):
        tracklet = old_df[n].dropna(axis=0)
        old_col1 = tracklet[col_name1]
        old_col2 = tracklet[col_name2]
        ind = old_col1.index
        new_col1 = []
        new_col2 = []
        for t in ind:
            new_col1.append(all_old2new_labels[t][int(old_col1[t])])
            new_col2.append(all_old2new_idx[t][int(old_col2[t])])

        old_df.loc[ind, (n, col_name1)] = new_col1
        old_df.loc[ind, (n, col_name2)] = new_col2

    return old_df, all_old2new_idx, all_old2new_labels


def _get_props(this_seg, this_img=None):
    props = regionprops(this_seg.copy(), intensity_image=this_img.copy())
    centroids = {}
    labels = {}
    for i, p in enumerate(props):
        centroids[i] = p.weighted_centroid
        labels[i] = p.label
    return centroids, labels
