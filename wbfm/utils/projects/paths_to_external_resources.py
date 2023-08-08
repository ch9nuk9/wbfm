import os
from pathlib import Path

import numpy as np

# As of May 2022
CAMERA_ALIGNMENT_MATRIX = "/scratch/neurobiology/zimmer/fieseler/repos/dlc_for_wbfm/wbfm/nn_checkpoints/" \
                          "warp_mat_green_to_red_2022-04-09.npy"


def get_precalculated_camera_alignment_matrix():
    if Path(CAMERA_ALIGNMENT_MATRIX).exists():
        return np.load(CAMERA_ALIGNMENT_MATRIX)
    else:
        return None


def get_pretrained_network_path(num_neurons):
    return os.path.join(PRETRAINED_DLC_DIR, f"{num_neurons}", "snapshot-100000")
