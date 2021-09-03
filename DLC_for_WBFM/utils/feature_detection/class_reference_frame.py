from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_cv2 import get_keypoints_from_3dseg
from DLC_for_WBFM.utils.feature_detection.custom_errors import OverwritePreviousAnalysisError, DataSynchronizationError, \
    AnalysisOutOfOrderError
from DLC_for_WBFM.utils.feature_detection.utils_detection import detect_neurons_from_file
from DLC_for_WBFM.utils.feature_detection.utils_features import convert_to_grayscale, detect_keypoints_and_features, \
    build_feature_tree, build_neuron_tree, build_f2n_map, detect_only_keypoints
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume

##
## Basic class definition
##

@dataclass
class ReferenceFrame:
    """ Information for registered reference frames"""

    # Data for registration
    neuron_locs: list = None
    keypoints: list = None
    keypoint_locs: np.ndarray = None  # Includes the z coordinate
    all_features: np.ndarray = None
    features_to_neurons: dict = None

    # Metadata
    frame_ind: int = None
    video_fname: str = None
    vol_shape: tuple = None

    preprocessing_settings: Union[PreprocessingSettings, None] = None

    # To be finished with a set of other registered frames
    neuron_ids: list = None  # global neuron index

    def get_metadata(self):
        return {'frame_ind': self.frame_ind,
                'video_fname': self.video_fname,
                'vol_shape': self.vol_shape}

    def iter_neurons(self):
        # Practice with yield
        for neuron in self.neuron_locs:
            yield neuron

    def get_features_of_neuron(self, which_neuron):
        iter_tmp = self.features_to_neurons.items()
        return [key for key, val in iter_tmp if val == which_neuron]

    def num_neurons(self):
        return self.neuron_locs.shape[0]

    def get_raw_data(self):
        return get_single_volume(self.video_fname,
                                 self.frame_ind,
                                 num_slices=self.vol_shape[0],
                                 alpha=self.preprocessing_settings.alpha)

    def detect_or_import_neurons(self, dat: list, external_detections: str, metadata: dict, num_slices: int,
                                 start_slice: int) -> list:
        """

        Parameters
        ----------
        dat
        external_detections
        metadata
        num_slices
        start_slice

        Returns
        -------
        neuron_locs - also saved as self.neuron_locs and self.keypoint_locs

        """
        if external_detections is None:
            from DLC_for_WBFM.utils.feature_detection.legacy_neuron_detection import detect_neurons_using_ICP
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

        if len(neuron_locs) == 0:
            print("No neurons detected... check data settings")
            # TODO: do not just raise an error, but instead skip rest of analysis
            raise ValueError

        self.neuron_locs = neuron_locs

        return neuron_locs

    def copy_neurons_to_keypoints(self):
        """ Explicitly a different method for backwards compatibility"""
        self.keypoint_locs = self.neuron_locs.copy()

    def detect_non_neuron_keypoints(self,
                                    dat,
                                    num_features_per_plane=1000,
                                    start_plane=0,
                                    append_to_existing_keypoints=False,
                                    verbose=0):
        """ See: detect_keypoints_and_build_features"""
        if not append_to_existing_keypoints:
            if self.keypoints is not None:
                raise OverwritePreviousAnalysisError('keypoints')

        all_locs = []
        all_kps = []
        for i in range(dat.shape[0]):
            if i < start_plane:
                continue
            im = np.squeeze(dat[i, ...])
            kp = detect_only_keypoints(im, num_features_per_plane)

            all_kps.extend(kp)
            locs_3d = np.array([np.hstack((i, row.pt)) for row in kp])
            all_locs.extend(locs_3d)

        all_locs = np.array(all_locs)

        if append_to_existing_keypoints:
            self.keypoints.append(all_kps)
            self.keypoint_locs = np.vstack([self.keypoint_locs, all_locs])
        else:
            self.keypoints = all_kps
            self.keypoint_locs = all_locs

        return all_kps, all_locs

    def detect_keypoints_and_build_features(self,
                                            dat,
                                            num_features_per_plane=1000,
                                            start_plane=0,
                                            verbose=0):
        if self.keypoints is not None:
            raise OverwritePreviousAnalysisError('keypoints')

        all_features = []
        all_locs = []
        all_kps = []
        for i in range(dat.shape[0]):
            if i < start_plane:
                continue
            im = np.squeeze(dat[i, ...])
            kp, features = detect_keypoints_and_features(im, num_features_per_plane)

            if features is None:
                continue
            all_features.extend(features)
            all_kps.extend(kp)
            locs_3d = np.array([np.hstack((i, row.pt)) for row in kp])
            all_locs.extend(locs_3d)

        all_locs, all_features = np.array(all_locs), np.array(all_features)

        self.keypoints = all_kps
        self.keypoint_locs = all_locs
        self.all_features = all_features

        return all_kps, all_locs, all_features

    def build_nontrivial_keypoint_to_neuron_mapping(self, neuron_feature_radius):
        """
        Matches keypoints and features based purely on distance (max=neuron_feature_radius)

        Designed when the keypoints are detected separately from the neurons
        Can also be used when the keypoints are a superset of neurons (e.g. ORB keypoints + neurons)

        """

        kp_3d_locs, neuron_locs = self.keypoint_locs, self.neuron_locs

        # Requires some open3d subfunctions; may not work on a cluster
        num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
        _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
        kp2n_map = build_f2n_map(kp_3d_locs,
                                 num_f,
                                 pc_f,
                                 neuron_feature_radius,
                                 tree_neurons,
                                 verbose=0)

        self.features_to_neurons = kp2n_map

        return kp2n_map

    def encode_all_keypoints(self, im_3d: np.ndarray, z_depth: int,
                             base_2d_encoder=None) -> Tuple[np.ndarray, list]:
        """
        Builds a feature vector for each neuron (zxy location) in a 3d volume
        Uses opencv VGG as a 2d encoder for a number of slices above and below the exact z location

        Note: overwrites the keypoints using only the locations

        Creates feature vectors of length z_depth *
        """

        locs_zxy = self.keypoint_locs

        im_3d_gray = [convert_to_grayscale(xy).astype('uint8') for xy in im_3d]
        all_embeddings = []
        all_keypoints = [None] * len(locs_zxy)
        if base_2d_encoder is None:
            base_2d_encoder = cv2.xfeatures2d.VGG_create()

        # Loop per plane, getting all keypoints for this plane
        for z in range(im_3d.shape[0]):
            # Slice band
            slices_around_keypoint = np.arange(z - z_depth, z + z_depth + 1)
            slices_around_keypoint = np.clip(slices_around_keypoint, 0, len(im_3d_gray) - 1)

            # Get all keypoints (not just this slice, but < z_depth away)
            these_locs_ind = np.where(np.abs(locs_zxy[:, 0] - z <= z_depth))[0]
            these_kp_2d = []

            # Embed all keypoints

            # Save


        for loc in tqdm(locs_zxy, leave=False):
            z, x, y = loc
            kp_2d = cv2.KeyPoint(x, y, 31.0)

            z = int(z)
            slices_around_keypoint = np.arange(z - z_depth, z + z_depth + 1)
            slices_around_keypoint = np.clip(slices_around_keypoint, 0, len(im_3d_gray) - 1)
            # Generate features on neighboring z slices as well
            # Repeat slices if near the edge
            one_kp_embedding = []
            for i in slices_around_keypoint:
                im_2d = im_3d_gray[int(i)]
                _, this_ds = base_2d_encoder.compute(im_2d, [kp_2d])
                one_kp_embedding.append(this_ds)

            one_kp_embedding = np.hstack(one_kp_embedding)
            all_embeddings.extend(one_kp_embedding)
            all_keypoints.append(kp_2d)

        all_embeddings = np.array(all_embeddings)

        self.all_features = all_embeddings
        self.keypoints = all_keypoints

        return all_embeddings, all_keypoints

    def build_trivial_keypoint_to_neuron_mapping(self):
        # This is now just a trivial mapping
        self.check_data_desyncing()
        kp2n_map = {i: i for i in range(len(self.neuron_locs))}
        self.features_to_neurons = kp2n_map
        return kp2n_map

    def prep_for_pickle(self):
        """Deletes the cv2.Keypoints (the locations are stored though)"""
        self.check_data_desyncing()
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

    def check_data_desyncing(self):
        # Keypoints and locations
        if len(self.keypoint_locs) != len(self.keypoints):
            raise DataSynchronizationError('keypoint_locs', 'keypoints', 'rebuild_keypoints')

        # Keypoints and features
        if len(self.keypoint) != len(self.all_features):
            raise DataSynchronizationError('all_features', 'keypoints')

    def __str__(self):
        return f"=======================================\n\
ReferenceFrame:\n\
Frame index: {self.frame_ind} \n\
Number of neurons: {len(self.neuron_locs)} \n\
Number of keypoints: {len(self.keypoint_locs)} \n"

    def __repr__(self):
        return f"ReferenceFrame with {len(self.neuron_locs)} neurons \n"


##
## Class for Set of reference frames
##

@dataclass
class RegisteredReferenceFrames():
    """Data for matched reference frames"""

    # Intermediate products
    reference_frames: list = None
    pairwise_matches: dict = None
    pairwise_conf: dict = None

    # More detailed intermediates and alternate matchings
    feature_matches: dict = None
    bipartite_matches: list = None

    # Global neuron coordinate system
    neuron_cluster_mode: str = None
    global2local: dict = None
    local2global: dict = None

    verbose: int = 0

    def __str__(self):
        return f"RegisteredReferenceFrames with {len(self.reference_frames)} Frames \n"

    def __repr__(self):
        [print(r) for r in self.reference_frames]
        return f"=======================================\n\
                RegisteredReferenceFrames:\n\
                Number of frames: {len(self.reference_frames)} \n"




def build_reference_frame_encoding(dat_raw,
                                   num_slices,
                                   z_depth,
                                   start_slice=None,
                                   metadata=None,
                                   external_detections=None,
                                   to_add_orb_keypoints=False,
                                   verbose=0):
    """
    New pipeline that directly builds an embedding for each neuron, instead of detecting keypoints

    See: build_reference_frame
    """
    if metadata is None:
        metadata = {}

    # Initialize class
    frame = ReferenceFrame(**metadata, preprocessing_settings=None)

    # Build keypoints (in this case, neurons directly)
    frame.detect_or_import_neurons(dat_raw, external_detections, metadata, num_slices, start_slice)
    frame.copy_neurons_to_keypoints()

    # Calculate encodings
    frame.encode_all_keypoints(dat_raw, z_depth)

    # Set up mapping between neurons and keypoints
    frame.build_trivial_keypoint_to_neuron_mapping()
    #
    #
    # dat = dat_raw
    # neuron_zxy = _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)
    #
    # embeddings, keypoints = encode_all_neurons(neuron_zxy, dat, z_depth)
    #
    # # This is now just a trivial mapping
    # f2n_map = {i: i for i in range(len(neuron_zxy))}
    # f = ReferenceFrame(neuron_zxy, keypoints, neuron_zxy, embeddings, f2n_map,
    #                    **metadata,
    #                    preprocessing_settings=None)

    return frame


def build_reference_frame(dat: np.ndarray,
                          num_slices: int,
                          neuron_feature_radius: float,
                          preprocessing_settings: PreprocessingSettings = None,
                          start_slice: int = 2,
                          metadata: dict = None,
                          external_detections: str = None,
                          verbose: int = 0) -> ReferenceFrame:
    """Main convenience constructor for ReferenceFrame class"""
    if metadata is None:
        metadata = {}

    # Initialize class
    frame = ReferenceFrame(**metadata, preprocessing_settings=None)

    # Build neurons and keypoints
    frame.detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)

    feature_opt = {'num_features_per_plane': 1000, 'start_plane': 5}
    frame.detect_keypoints_and_build_features(dat, **feature_opt)

    # Set up mapping between neurons and keypoints
    frame.build_trivial_keypoint_to_neuron_mapping(neuron_feature_radius)

    # Get neurons and features, and a map between them
    # neuron_locs = _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)
    # if len(neuron_locs) == 0:
    #     print("No neurons detected... check data settings")
    #     raise ValueError
    # kps, kp_3d_locs, features = build_features_1volume(dat, **feature_opt)
    #
    # # The map requires some open3d subfunctions; may not work on a cluster
    # num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
    # _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
    # f2n_map = build_f2n_map(kp_3d_locs,
    #                         num_f,
    #                         pc_f,
    #                         neuron_feature_radius,
    #                         tree_neurons,
    #                         verbose=verbose - 1)
    #
    # # Finally, my summary class
    # f = ReferenceFrame(neuron_locs, kps, kp_3d_locs, features, f2n_map,
    #                    **metadata,
    #                    preprocessing_settings=preprocessing_settings)
    return f