import numpy as np
from DLC_for_WBFM.utils.external.centerline_utils import transform_neuron_point_cloud
from DLC_for_WBFM.utils.external.dino import PointCloudAugmentationDINO
from DLC_for_WBFM.utils.postures.centerline_pca import WormSinglePosture
from tqdm.auto import tqdm
import torch


def build_worm_at_time(t, project_data, worm_posture):
    pts0 = transform_neuron_point_cloud(project_data.get_centroids_as_numpy(t))
    centerline0 = worm_posture.get_centerline_for_time(t)
    return WormSinglePosture(pts0, centerline0)


# ============ preparing data ... ============
# n_neighbors = dim_h = 150
# dim_w = 3
# batch_sz = 1
# student_crops_number = 4
# use_all_points = True
#
# augmentation = PointCloudAugmentationDINO(teacher_num_to_replace=0, student_num_to_replace=1,
#                                           student_crops_number=student_crops_number)


class PointCloudDataset():

    def __init__(self, augmentation, n_neighbors,
                 project_data, worm_posture,
                 num_volumes=1, start_volume=0, use_all_points=False):

        preloaded_data = []
        for t in tqdm(range(start_volume, start_volume + num_volumes)):
            worm = build_worm_at_time(t, project_data, worm_posture)
            try:
                for i_anchor in range(len(worm.neuron_zxy)):
                    if use_all_points:
                        pts = worm.get_all_neurons_in_local_coordinate_system(i_anchor=i_anchor)
                        if pts.shape[0] == n_neighbors:
                            continue
                        elif pts.shape[0] > n_neighbors:
                            # print("Taking subset of neighbors")
                            pts = pts[:n_neighbors, :]
                        else:
                            to_pad = n_neighbors - pts.shape[0]
                            pts = np.pad(pts, ((0, to_pad), (0, 0)), constant_values=0.0)
                    else:
                        pts = worm.get_neighbors_in_local_coordinate_system(i_anchor=i_anchor, n_neighbors=n_neighbors)
                    preloaded_data.append(torch.from_numpy(np.expand_dims(pts, axis=0)))
            except IndexError:
                print("skipped an index error")
                pass

        self.preloaded_data = preloaded_data

        self.preaugmented_data = [(augmentation(dat), None) for dat in preloaded_data]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, idx):
        return self.preaugmented_data[idx]  # Already includes "label", i.e. None
        # return self.augmentation(self.preloaded_data[idx]), None
