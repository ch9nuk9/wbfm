import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from DLC_for_WBFM.utils.external.utils_networkx import calc_bipartite_from_positions
from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from skimage.measure import regionprops
from tqdm.auto import tqdm
from segmentation.util.utils_metadata import DetectedNeurons



def remap_tracklets_to_new_segmentation(project_data: ProjectData,
                                        path_to_new_segmentation,
                                        path_to_new_metadata,
                                        DEBUG=False):

    new_meta = DetectedNeurons(path_to_new_metadata)
    print(new_meta)
    old_seg = project_data.raw_segmentation
    new_seg = zarr.open(path_to_new_segmentation)
    red = project_data.red_data
    num_frames = project_data.num_frames
    old_df = project_data.df_all_tracklets
    try:
        logging.info("Converting dataframe to dense form, may take a while")
        if DEBUG:
            new_df = old_df[:5].sparse.to_dense()
        else:
            new_df = old_df.sparse.to_dense()
    except AttributeError:
        new_df = old_df.copy()
    names = get_names_from_df(old_df)

    if DEBUG:
        num_frames = 5

    all_old2new_idx, all_old2new_labels = match_two_segmentations(new_seg, num_frames, old_seg, red)

    logging.info("Updating tracklet-segmentation indices using mapping")
    col_name1 = 'raw_segmentation_id'
    col_name2 = 'raw_neuron_ind_in_list'
    for n in tqdm(names):
        tracklet = old_df[n].dropna(axis=0)
        ind, new_col1, new_col2 = get_new_column_values_using_mapping(all_old2new_idx, all_old2new_labels, col_name1,
                                                                      col_name2, num_frames, tracklet)

        new_df.loc[ind, (n, col_name1)] = new_col1
        new_df.loc[ind, (n, col_name2)] = new_col2

    # Note; this loop could be combined with above if needed
    logging.info("Updating metadata using new indices")
    cols_to_replace = ['z', 'x', 'y', 'brightness_red', 'volume']
    for n in tqdm(names):
        new_columns = defaultdict(list)
        tracklet = new_df[n].dropna(axis=0)
        ind = tracklet.index
        ind = ind[ind < num_frames]

        for t in ind:
            mask_ind = tracklet.at[t, 'raw_segmentation_id']
            row_data, column_names = new_meta.get_all_metadata_for_single_time(mask_ind, t, None)

            # Only need certain columns
            for val, col_name in zip(row_data, column_names):
                if col_name in cols_to_replace:
                    new_columns[col_name].append(val)
        # Update all columns at once
        for col_name in cols_to_replace:
            new_df.loc[ind, (n, col_name)] = new_columns[col_name]

    _save_new_tracklets_and_update_config_file(new_df, path_to_new_metadata, path_to_new_segmentation, project_data)

    return new_df, all_old2new_idx, all_old2new_labels


def _save_new_tracklets_and_update_config_file(new_df, path_to_new_metadata, path_to_new_segmentation, project_data):
    # Save
    logging.info("Saving")
    track_cfg = project_data.project_config.get_tracking_config()
    df_to_save = new_df.astype(pd.SparseDtype("float", np.nan))
    output_df_fname = os.path.join('3-tracking', 'postprocessing', 'df_resegmented.pickle')
    track_cfg.pickle_in_local_project(df_to_save, output_df_fname, custom_writer=pd.to_pickle)
    # logging.warning("Overwriting name of manual correction tracklets, assuming that was the most recent")
    df_fname = track_cfg.unresolve_absolute_path(output_df_fname)
    track_cfg.config.update({'manual_correction_tracklets_df_fname': df_fname})
    track_cfg.update_on_disk()
    segmentation_cfg = project_data.project_config.get_segmentation_config()
    fname = segmentation_cfg.unresolve_absolute_path(path_to_new_segmentation)
    segmentation_cfg.config['output_masks'] = fname
    fname = segmentation_cfg.unresolve_absolute_path(path_to_new_metadata)
    segmentation_cfg.config['output_metadata'] = fname
    segmentation_cfg.update_on_disk()


def match_two_segmentations(new_seg, num_frames, old_seg, red):
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
    return all_old2new_idx, all_old2new_labels


def get_new_column_values_using_mapping(all_old2new_idx, all_old2new_labels, col_name1, col_name2, num_frames,
                                        tracklet):
    old_col1 = tracklet[col_name1]
    old_col2 = tracklet[col_name2]
    ind = old_col1.index
    ind = ind[ind < num_frames]
    new_col1 = []
    new_col2 = []
    for t in ind:
        if t >= num_frames:
            break
        new_col1.append(all_old2new_labels[t][int(old_col1[t])])
        new_col2.append(all_old2new_idx[t][int(old_col2[t])])
    return ind, new_col1, new_col2


def _get_props(this_seg, this_img=None):
    props = regionprops(this_seg.copy(), intensity_image=this_img.copy())
    centroids = {}
    labels = {}
    for i, p in enumerate(props):
        centroids[i] = p.weighted_centroid
        labels[i] = p.label
    return centroids, labels


def remap_tracklets_to_new_segmentation_using_config(project_path: str,
                                                     path_to_new_segmentation,
                                                     path_to_new_metadata,
                                                     DEBUG=False):
    project_data = ProjectData.load_final_project_data_from_config(project_path)

    remap_tracklets_to_new_segmentation(project_data,
                                        path_to_new_segmentation,
                                        path_to_new_metadata,
                                        DEBUG)
