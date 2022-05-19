import os
import numpy as np

# As of May 2022
CAMERA_ALIGNMENT_MATRIX = "/scratch/neurobiology/zimmer/Charles/repos/dlc_for_wbfm/DLC_for_WBFM/nn_checkpoints/" \
                          "warp_mat_green_to_red_2022-04-09.npy"


def get_camera_alignment_matrix():
    return np.load(CAMERA_ALIGNMENT_MATRIX)


def get_pretrained_network_path(num_neurons):
    return os.path.join(PRETRAINED_DLC_DIR, f"{num_neurons}", "snapshot-100000")
