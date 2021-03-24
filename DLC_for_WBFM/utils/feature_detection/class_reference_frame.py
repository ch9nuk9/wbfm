from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.external.utils_cv2 import get_keypoints_from_3dseg
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from dataclasses import dataclass
from typing import List
import numpy as np


##
## Basic class definition
##

@dataclass
class ReferenceFrame():
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
        return [key for key,val in iter_tmp if val == which_neuron]
        #return np.argwhere(self.features_to_neurons == which_neuron)

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
        return f"Number of neurons: {len(self.neuron_locs)} \n"

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
