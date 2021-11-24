import numpy as np
import pandas as pd
from skimage.measure import regionprops
import torch


def get_bbox_data(i_tracklet, df, project_data, t_local=None, target_sz=np.array([8, 64, 64])):
    """if t_local is None, chooses a random time"""
    # Note that .at properly selects the row by index even if some rows are dropped
    track_seg_ind = df.at[i_tracklet, 'all_ind_local']
    track_time_ind = df.at[i_tracklet, 'slice_ind']

    if t_local is None:
        t_max = len(track_seg_ind)
        t_local = np.random.randint(0, t_max)

    t_global = track_time_ind[t_local]
    i_local = track_seg_ind[t_local]

    seg_local = project_data.segmentation_metadata.seg_array_to_mask_index(t_global, i_local)

    # Get a bbox for a neuron in 3d
    this_seg = project_data.raw_segmentation[t_global, ...]
    props = regionprops(this_seg)

    p = [p for p in props if p.label == seg_local][0]

    bbox = p.bbox

    # Expand to get the neighborhood
    sz = project_data.red_data.shape

    z0 = np.clip(bbox[0] - 3, a_min=0, a_max=sz[1])
    z1 = np.clip(bbox[3] + 3, a_min=0, a_max=sz[1])
    if z1 - z0 > target_sz[0]:
        z1 = z0 + target_sz[0]
    x0 = np.clip(bbox[1] - 30, a_min=0, a_max=sz[2])
    x1 = np.clip(bbox[4] + 30, a_min=0, a_max=sz[2])
    if x1 - x0 > target_sz[1]:
        x1 = x0 + target_sz[1]
    y0 = np.clip(bbox[2] - 30, a_min=0, a_max=sz[3])
    y1 = np.clip(bbox[5] + 30, a_min=0, a_max=sz[3])
    if y1 - y0 > target_sz[2]:
        y1 = y0 + target_sz[2]

    dat = project_data.red_data[t_global, z0:z1, x0:x1, y0:y1]

    # Pad, if needed, to the beginning
    diff_sz = np.clip(target_sz - np.array(dat.shape), a_min=0, a_max=np.max(target_sz))
    pad_sz = list(zip(diff_sz, np.zeros(len(diff_sz), dtype=int)))
    dat = np.pad(dat, pad_sz)

    return dat, bbox


# MAX_TRACKLET = df.shape[0]
def get_siamese_training_triplet(df: pd.DataFrame, project_data):
    rng = np.random.default_rng()
    rand_order = rng.permutation(list(df.index))

    # Two examples from same, one different
    i_tracklet0 = rand_order[0]
    dat_anchor, _ = get_bbox_data(i_tracklet0, df, project_data, t_local=None)
    dat_pos, _ = get_bbox_data(i_tracklet0, df, project_data, t_local=None)

    i_tracklet1 = rand_order[1]
    dat_neg, _ = get_bbox_data(i_tracklet1, df, project_data, t_local=None)

    return dat_anchor, dat_pos, i_tracklet0, dat_neg, i_tracklet1


def get_one_triplet(df, project_data):
    d0, d1, label1, d2, label2 = get_siamese_training_triplet(df, project_data)
    d0, d1, d2 = torch.from_numpy(d0.astype(np.uint8)), torch.from_numpy(d1.astype(np.uint8)), torch.from_numpy(
        d2.astype(np.uint8))
    d0 = torch.unsqueeze(d0, 0)
    d0 = torch.unsqueeze(d0, 0).float() / 255.0
    d1 = torch.unsqueeze(d1, 0)
    d1 = torch.unsqueeze(d1, 0).float() / 255.0
    d2 = torch.unsqueeze(d2, 0)
    d2 = torch.unsqueeze(d2, 0).float() / 255.0
    return d0, d1, label1, d2, label2


def build_train_loader(df, project_data, max_iters=100):
    for i in range(max_iters):
        yield get_one_triplet(df, project_data)


def build_train_loader_batch(df, project_data, max_iters=100, batch_size=4):
    for i in range(max_iters):
        all_d0, all_d1, all_d2 = [], [], []
        for i2 in range(batch_size):
            d0, d1, label1, d2, label2 = get_one_triplet(df, project_data)
            all_d0.append(d0)
            all_d1.append(d1)
            all_d2.append(d2)
        yield torch.cat(all_d0, 0), torch.cat(all_d1, 0), None, torch.cat(all_d2, 0), None
