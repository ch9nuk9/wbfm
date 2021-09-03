from dataclasses import dataclass

from DLC_for_WBFM.utils.external.utils_cv2 import cast_matches_as_array
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.feature_detection.utils_affine import calc_matches_using_affine_propagation
from DLC_for_WBFM.utils.feature_detection.utils_features import match_known_features
from DLC_for_WBFM.utils.feature_detection.utils_gaussian_process import calc_matches_using_gaussian_process
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_candidates


@dataclass
class FramePair:
    """Information connecting neurons in two ReferenceFrame objects"""

    # Final output, with confidences
    final_matches: list

    # Intermediate products, with confidences
    feature_matches: list = None
    affine_matches: list = None
    affine_pushed_locations: list = None
    gp_matches: list = None
    gp_pushed_locations: list = None

    all_gps: list = None  # The actual gaussian processes; may get warning from scipy versioning

    # Original keypoints
    keypoint_matches: list = None

    min_confidence: float = 0.001

    # Frame classes
    frame0: ReferenceFrame = None
    frame1: ReferenceFrame = None

    @property
    def all_candidate_matches(self):
        all_matches = self.feature_matches.copy()
        if self.affine_matches is not None:
            all_matches.extend(self.affine_matches)
        if self.gp_matches is not None:
            all_matches.extend(self.gp_matches)
        return all_matches

    def calc_final_matches_using_bipartite_matching(self) -> list:
        assert len(self.all_candidate_matches) > 0, "No candidate matches!"

        matches, conf, _ = calc_bipartite_from_candidates(self.all_candidate_matches, min_conf=self.min_confidence)
        self.final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]
        return self.final_matches

    def get_f0_to_f1_dict(self):
        return {n0: n1 for n0, n1, _ in self.final_matches}

    def get_f1_to_f0_dict(self):
        return {n1: n0 for n0, n1, _ in self.final_matches}

    def calculate_additional_orb_keypoints_and_matches(self):
        return False

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)} matches \n"


def calc_FramePair_from_Frames(frame0: ReferenceFrame,
                               frame1: ReferenceFrame,
                               verbose: int = 1,
                               add_affine_to_candidates: bool = False,
                               add_gp_to_candidates: bool = False,
                               DEBUG: bool = False) -> FramePair:
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    Main constructor for the class FramePair

    See also: calc_2frame_matches
    """

    frame0.check_data_desyncing()
    frame1.check_data_desyncing()

    # First, get feature matches
    keypoint_matches = match_known_features(frame0.all_features,
                                            frame1.all_features,
                                            frame0.keypoints,
                                            frame1.keypoints,
                                            frame0.vol_shape[1:],
                                            frame1.vol_shape[1:],
                                            matches_to_keep=1.0,
                                            use_GMS=False)

    # With neuron embeddings, the keypoints are the neurons
    matches_with_conf = cast_matches_as_array(keypoint_matches, gamma=1.0)

    # Create convenience object to store matches
    frame_pair = FramePair(matches_with_conf, matches_with_conf,
                           frame0=frame0, frame1=frame1)
    frame_pair.keypoint_matches = matches_with_conf

    # Add additional candidates, if used
    if add_affine_to_candidates:
        options = {'all_feature_matches': keypoint_matches}
        matches_with_conf, _, affine_pushed = calc_matches_using_affine_propagation(frame0, frame1, **options)
        frame_pair.affine_matches = matches_with_conf
        frame_pair.affine_pushed_locations = affine_pushed

    if add_gp_to_candidates:
        n0 = frame0.neuron_locs.copy()
        n1 = frame1.neuron_locs.copy()

        # TODO: Increase z distances correctly
        n0[:, 0] *= 5
        n1[:, 0] *= 5
        # Actually match
        options = {'matches_with_conf': matches_with_conf}
        matches_with_conf, all_gps, gp_pushed = calc_matches_using_gaussian_process(n0, n1, **options)
        frame_pair.gp_matches = matches_with_conf
        frame_pair.all_gps = all_gps
        frame_pair.gp_pushed_locations = gp_pushed

    return frame_pair
