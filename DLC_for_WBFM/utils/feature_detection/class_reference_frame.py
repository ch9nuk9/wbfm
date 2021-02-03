from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from dataclasses import dataclass


##
## Class to hold preprocessing settings
##

@dataclass
class PreprocessingSettings():
    """
    Holds settings that will be applied to the ReferenceFrame class
    """

    # Filtering
    do_filtering : bool = False
    filter_opt : dict = {'high_freq':2.0, 'low_freq':5000.0}

    # Mini max
    do_mini_max_projection : bool = False
    mini_max_size : int = 3

    # Rigid alignment (slices to each other)
    do_rigid_alignment : bool = False


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
    alpha: float = 1.0

    preprocessing_settings: PreprocessingSettings = None

    # To be finished with a set of other registered frames
    neuron_ids: list = None # global neuron index

    def get_metadata(self):
        return {'frame_ind':self.frame_ind,
                'video_fname':self.video_fname,
                'vol_shape':self.vol_shape,
                'alpha':self.alpha}

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
                                 alpha=self.alpha)

    def __str__(self):
        return f"=======================================\n\
                ReferenceFrame:\n\
                Frame index: {self.frame_ind} \n\
                Number of neurons: {len(self.neuron_locs)} \n"

##
## Class for Set of reference frames
##

@dataclass
class RegisteredReferenceFrames():
    """Data for matched reference frames"""

    # Global neuron coordinate system
    global2local : dict
    local2global : dict

    # Intermediate products
    reference_frames : list = None
    pairwise_matches : dict = None
    pairwise_conf : dict = None

    # More detailed intermediates and alternate matchings
    feature_matches : dict = None
    bipartite_matches : list = None

    def __str__(self):
        [print(r) for r in self.reference_frames]
        return f"=======================================\n\
                RegisteredReferenceFrames:\n\
                Number of frames: {len(self.reference_frames)} \n"
