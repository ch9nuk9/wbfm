import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

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

    # Metadata
    add_affine_to_candidates: bool = False
    add_gp_to_candidates: bool = False

    @property
    def all_candidate_matches(self):
        all_matches = self.feature_matches.copy()
        if self.affine_matches is not None:
            all_matches.extend(self.affine_matches)
        if self.gp_matches is not None:
            all_matches.extend(self.gp_matches)
        return all_matches

    @property
    def num_possible_matches(self):
        if self.frame0 is None:
            return np.nan
        return min(self.frame0.num_neurons(), self.frame1.num_neurons())

    def prep_for_pickle(self):
        self.frame0.prep_for_pickle()
        self.frame1.prep_for_pickle()

    def rebuild_keypoints(self):
        self.frame0.rebuild_keypoints()
        self.frame1.rebuild_keypoints()

    def calc_final_matches_using_bipartite_matching(self, min_confidence: float = None,
                                                    z_threshold=None) -> list:
        assert len(self.all_candidate_matches) > 0, "No candidate matches!"
        if min_confidence is None:
            min_confidence = self.min_confidence
        else:
            self.min_confidence = min_confidence

        matches, conf, _ = calc_bipartite_from_candidates(self.all_candidate_matches, min_conf=min_confidence)
        final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]
        final_matches = self.filter_matches_using_z_threshold(final_matches, z_threshold)
        self.final_matches = final_matches
        return final_matches

    def get_f0_to_f1_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {n0: n1 for n0, n1, _ in matches}

    def get_f1_to_f0_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {n1: n0 for n0, n1, _ in matches}

    def get_pair_to_conf_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {(n0, n1): c for n0, n1, c in matches}

    # def calculate_additional_orb_keypoints_and_matches(self):
    #     return False

    def get_metadata_dict(self):
        return {'add_affine_to_candidates': self.add_affine_to_candidates,
                'add_gp_to_candidates': self.add_gp_to_candidates}

    def save_matches_as_excel(self, target_dir='.'):
        f0_ind = self.frame0.frame_ind
        f1_ind = self.frame1.frame_ind
        df = pd.DataFrame(self.final_matches, columns=[f'Neuron_in_f{f0_ind}', f'Neuron_in_f{f1_ind}', 'Confidence'])
        fname = f'matches_f{f0_ind}_f{f1_ind}.xlsx'
        fname = os.path.join(target_dir, fname)
        df.to_excel(fname, index=False)

    def filter_matches_using_z_threshold(self, matches, z_threshold):
        if z_threshold is None:
            return matches
        n0 = self.frame0.neuron_locs.copy()
        n1 = self.frame1.neuron_locs.copy()

        def _delta_z(m):
            return np.abs(n0[m[0]][0] - n1[m[1]][0])

        return [m for m in matches if _delta_z(m) < z_threshold]

    def print_candidates_by_method(self):
        num_matches = len(self.feature_matches)
        print(f"Found {num_matches} candidates via feature matching")
        num_matches = len(self.affine_matches)
        print(f"Found {num_matches} candidates via affine matching")
        num_matches = len(self.gp_matches)
        print(f"Found {num_matches} candidates via gaussian process matching")

        num_matches = len(self.final_matches)
        print(f"Processed these into {num_matches} final matches candidates")

    def print_reason_for_match(self, test_match):
        m0, m1 = test_match

        f0_to_1 = self.get_f0_to_f1_dict()
        if f0_to_1[m0] == m1:
            f_to_conf = self.get_pair_to_conf_dict()
            print(f"Found match {test_match} with confidence {f_to_conf[test_match]}")

            feature_dict = self.get_f0_to_f1_dict(self.feature_matches)
            method_name = "feature"
            if m0 in feature_dict:
                if feature_dict[m0] == m1:
                    conf = self.get_pair_to_conf_dict(self.feature_matches)[test_match]
                    print(f"Same match from {method_name} method with confidence: {conf}")
                else:
                    print(f"Different match from {method_name} method: {feature_dict[m0]}")
            else:
                print(f"Neuron not matched using {method_name} method")

            aff_dict = self.get_f0_to_f1_dict(self.affine_matches)
            method_name = "affine"
            if m0 in aff_dict:
                if aff_dict[m0] == m1:
                    conf = self.get_pair_to_conf_dict(self.affine_matches)[test_match]
                    print(f"Same match from {method_name} method with confidence: {conf}")
                else:
                    print(f"Different match from {method_name} method: {aff_dict[m0]}")
            else:
                print(f"Neuron not matched using {method_name} method")

            gp_dict = self.get_f0_to_f1_dict(self.gp_matches)
            method_name = "gaussian process"
            if m0 in gp_dict:
                if gp_dict[m0] == m1:
                    conf = self.get_pair_to_conf_dict(self.gp_matches)[test_match]
                    print(f"Same match from {method_name} method with confidence: {conf}")
                else:
                    print(f"Different match from {method_name} method: {gp_dict[m0]}")
            else:
                print(f"Neuron not matched using {method_name} method")

    def print_reason_for_all_final_matches(self):
        dict_of_matches = self.get_pair_to_conf_dict()
        for k in dict_of_matches.keys():
            print("==================================")
            self.print_reason_for_match(k)

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)}/{self.num_possible_matches} matches \n"


def calc_FramePair_from_Frames(frame0: ReferenceFrame,
                               frame1: ReferenceFrame,
                               verbose: int = 1,
                               add_affine_to_candidates: bool = False,
                               add_gp_to_candidates: bool = False,
                               min_confidence: float = 0.001,
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
                           frame0=frame0, frame1=frame1,
                           add_affine_to_candidates=add_affine_to_candidates,
                           add_gp_to_candidates=add_gp_to_candidates,
                           min_confidence=min_confidence)
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
        n0[:, 0] *= 3
        n1[:, 0] *= 3
        # Actually match
        options = {'matches_with_conf': matches_with_conf}
        matches_with_conf, all_gps, gp_pushed = calc_matches_using_gaussian_process(n0, n1, **options)
        frame_pair.gp_matches = matches_with_conf
        frame_pair.all_gps = all_gps
        frame_pair.gp_pushed_locations = gp_pushed

    return frame_pair


def calc_FramePair_like(pair: FramePair, frame0: ReferenceFrame = None, frame1: ReferenceFrame = None) -> FramePair:
    """
    Calculates a new frame pair using the metadata from the FramePair, and new frames (optional)

    Parameters
    ----------
    pair
    frame0
    frame1

    Returns
    -------

    """

    metadata = pair.get_metadata_dict()

    if frame0 is None:
        frame0 = pair.frame0
    if frame1 is None:
        frame1 = pair.frame1

    new_pair = calc_FramePair_from_Frames(frame0, frame1, **metadata)
    new_pair.calc_final_matches_using_bipartite_matching(pair.min_confidence)

    return new_pair
