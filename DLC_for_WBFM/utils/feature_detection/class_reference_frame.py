from typing import Tuple

import cv2

from DLC_for_WBFM.utils.feature_detection.utils_detection import detect_neurons_using_ICP, detect_neurons_from_file
from DLC_for_WBFM.utils.feature_detection.utils_features import convert_to_grayscale
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.external.utils_cv2 import get_keypoints_from_3dseg
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from dataclasses import dataclass
import numpy as np


##
## Basic class definition
##

@dataclass
class ReferenceFrame:
    """ Information for registered reference frames"""

    # Data for registration
    neuron_locs: list
    keypoints: list
    keypoint_locs: list # Includes the z coordinate
    all_features: np.array
    features_to_neurons: dict

    # Metadata
    frame_ind: int = None
    video_fname: str = None
    vol_shape: tuple = None

    preprocessing_settings: PreprocessingSettings = None

    # To be finished with a set of other registered frames
    neuron_ids: list = None # global neuron index

    def get_metadata(self):
        return {'frame_ind':self.frame_ind,
                'video_fname':self.video_fname,
                'vol_shape':self.vol_shape}

    def iter_neurons(self):
        # Practice with yield
        for neuron in self.neuron_locs:
            yield neuron

    def get_features_of_neuron(self, which_neuron):
        iter_tmp = self.features_to_neurons.items()
        return [key for key, val in iter_tmp if val == which_neuron]

    def num_neurons(self):
        return self.neuron_locs.shape[0]

    def get_data(self):
        return get_single_volume(self.video_fname,
                                 self.frame_ind,
                                 num_slices=self.vol_shape[0],
                                 alpha=self.preprocessing_settings.alpha)

    def prep_for_pickle(self):
        """Deletes the cv2.Keypoints (the locations are stored though)"""
        self.keypoints = []



    def rebuild_keypoints(self):
        """
        Rebuilds keypoints from keypoint_locs
        see also self.prep_for_pickle()
        """
        if len(self.keypoints) > 0:
            print("Overwriting existing keypoints...")
        k = get_keypoints_from_3dseg(self.keypoint_locs)
        self.keypoints = k


    def __str__(self):
        return f"=======================================\n\
                ReferenceFrame:\n\
                Frame index: {self.frame_ind} \n\
                Number of neurons: {len(self.neuron_locs)} \n"

    def __repr__(self):
        return f"ReferenceFrame with {len(self.neuron_locs)} neurons \n"

##
## Class for Set of reference frames
##

@dataclass
class RegisteredReferenceFrames():
    """Data for matched reference frames"""

    # Intermediate products
    reference_frames : list = None
    pairwise_matches : dict = None
    pairwise_conf : dict = None

    # More detailed intermediates and alternate matchings
    feature_matches : dict = None
    bipartite_matches : list = None

    # Global neuron coordinate system
    neuron_cluster_mode : str = None
    global2local : dict = None
    local2global : dict = None

    verbose : int = 0

    def __str__(self):
        return f"RegisteredReferenceFrames with {len(self.reference_frames)} Frames \n"

    def __repr__(self):
        [print(r) for r in self.reference_frames]
        return f"=======================================\n\
                RegisteredReferenceFrames:\n\
                Number of frames: {len(self.reference_frames)} \n"


def _detect_or_import_neurons(dat: list, external_detections: str, metadata: dict, num_slices: int,
                              start_slice: int) -> list:
    if external_detections is None:
        neuron_locs, _, _, _ = detect_neurons_using_ICP(dat,
                                                        num_slices=num_slices,
                                                        alpha=1.0,
                                                        min_detections=3,
                                                        start_slice=start_slice,
                                                        verbose=0)
        neuron_locs = np.array([n for n in neuron_locs])
    else:
        i = metadata['frame_ind']
        neuron_locs = detect_neurons_from_file(external_detections, i)
    return neuron_locs


def build_reference_frame_encoding(dat_raw,
                                   num_slices,
                                   z_depth,
                                   start_slice=None,
                                   metadata=None,
                                   external_detections=None,
                                   verbose=0):
    """
    New pipeline that directly builds an embedding for each neuron, instead of detecting keypoints

    See: build_reference_frame
    """
    if metadata is None:
        metadata = {}
    dat = dat_raw
    neuron_zxy = _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)

    embeddings, keypoints = encode_all_neurons(neuron_zxy, dat, z_depth)

    # This is now just a trivial mapping
    f2n_map = {i: i for i in range(len(neuron_zxy))}
    f = ReferenceFrame(neuron_zxy, keypoints, neuron_zxy, embeddings, f2n_map,
                       **metadata,
                       preprocessing_settings=None)

    return f


def encode_all_neurons(locs_zxy: list, im_3d: np.ndarray, z_depth: int) -> Tuple[np.ndarray, list]:
    """
    Builds a feature vector for each neuron (zxy location) in a 3d volume
    Uses opencv VGG as a 2d encoder for a number of slices above and below the exact z location
    """
    im_3d_gray = [convert_to_grayscale(xy) for xy in im_3d]
    all_embeddings = []
    all_keypoints = []
    encoder = cv2.xfeatures2d.VGG_create()

    # Loop per neuron
    for loc in locs_zxy:
        z, x, y = loc
        kp = cv2.KeyPoint(x, y, 31.0)

        z = int(z)
        all_slices = np.arange(z - z_depth, z + z_depth + 1)
        all_slices = np.clip(all_slices, 0, len(im_3d_gray)-1)
        # Generate features on neighboring z slices as well
        # Repeat slices if near the edge
        ds = []
        for i in all_slices:
            im_2d = im_3d_gray[int(i)].astype('uint8')
            _, this_ds = encoder.compute(im_2d, [kp])
            ds.append(this_ds)

        ds = np.hstack(ds)
        all_embeddings.extend(ds)
        all_keypoints.append(kp)

    return np.array(all_embeddings), all_keypoints