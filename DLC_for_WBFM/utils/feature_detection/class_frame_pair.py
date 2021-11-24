import os
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from segmentation.util.utils_metadata import DetectedNeurons
from DLC_for_WBFM.utils.external.utils_cv2 import cast_matches_as_array
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.feature_detection.custom_errors import NoMatchesError
from DLC_for_WBFM.utils.feature_detection.utils_affine import calc_matches_using_affine_propagation
from DLC_for_WBFM.utils.feature_detection.utils_features import match_known_features, build_features_and_match_2volumes
from DLC_for_WBFM.utils.feature_detection.utils_gaussian_process import calc_matches_using_gaussian_process
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_candidates, dist2conf
from DLC_for_WBFM.utils.nn_utils.data_formatting import zimmer2leifer, flatten_nested_list


@dataclass
class FramePairOptions:
    # Flag and options for each method
    # First: default feature-embedding method
    embedding_matches_to_keep: float = 1.0
    embedding_use_GMS: bool = False
    crossCheck: bool = True

    add_affine_to_candidates: bool = False
    start_plane: int = 4
    num_features_per_plane: int = 10000
    affine_matches_to_keep: float = 0.8
    affine_use_GMS: bool = True
    min_matches: int = 20
    allow_z_change: bool = False
    affine_num_candidates: int = 1

    add_gp_to_candidates: bool = False
    starting_matches: str = 'affine_matches'
    gp_num_candidates: int = 1

    add_fdnc_to_candidates: bool = False
    fdnc_options: dict = None

    # For filtering / postprocessing the matches
    z_threshold: float = None
    min_confidence: float = 0.001
    z_to_xy_ratio: float = 3.0

    def __post_init__(self):
        from DLC_for_WBFM.utils.nn_utils.fdnc_predict import load_fdnc_options
        default_options = load_fdnc_options()

        if self.fdnc_options is None:
            self.fdnc_options = {}
        else:
            default_options.update(self.fdnc_options)
        self.fdnc_options = default_options


@dataclass
class FramePair:
    """Information connecting neurons in two ReferenceFrame objects"""
    options: FramePairOptions = None

    # Final output, with confidences
    final_matches: list = None

    # Intermediate products, with confidences
    feature_matches: list = None
    affine_matches: list = None
    affine_pushed_locations: list = None
    gp_matches: list = None
    gp_pushed_locations: list = None

    all_gps: list = None  # The actual gaussian processes; may get warning from scipy versioning

    # New method:
    fdnc_matches: list = None

    # Original keypoints
    keypoint_matches: list = None

    # Frame classes
    frame0: ReferenceFrame = None
    frame1: ReferenceFrame = None

    @property
    def all_candidate_matches(self) -> list:
        all_matches = self.feature_matches.copy()
        if self.options.add_affine_to_candidates:
            all_matches.extend(self.affine_matches)
        if self.options.add_gp_to_candidates:
            all_matches.extend(self.gp_matches)
        if self.options.add_fdnc_to_candidates:
            all_matches.extend(self.fdnc_matches)
        return all_matches

    @property
    def num_possible_matches(self) -> int:
        if self.frame0 is None:
            return np.nan
        return min(self.frame0.num_neurons(), self.frame1.num_neurons())

    def prep_for_pickle(self):
        self.frame0.prep_for_pickle()
        self.frame1.prep_for_pickle()

    def rebuild_keypoints(self):
        self.frame0.rebuild_keypoints()
        self.frame1.rebuild_keypoints()

    def calc_final_matches(self, method='bipartite', **kwargs):
        if method == 'bipartite':
            return self.calc_final_matches_using_bipartite_matching(**kwargs)
        elif method == 'unanimous':
            return self.calc_final_matches_using_unanimous_voting(**kwargs)
        else:
            raise NotImplementedError

    def calc_final_matches_using_bipartite_matching(self, min_confidence: float = None,
                                                    z_threshold=None) -> list:
        assert len(self.all_candidate_matches) > 0, "No candidate matches!"
        z_threshold, min_confidence = self.use_defaults_if_none(min_confidence, z_threshold)

        try:
            matches, conf, _ = calc_bipartite_from_candidates(self.all_candidate_matches,
                                                              min_confidence_after_sum=min_confidence)
            final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]
            final_matches = self.filter_matches_using_z_threshold(final_matches, z_threshold)
        except NoMatchesError:
            final_matches = []
        self.final_matches = final_matches
        return final_matches

    def calc_final_matches_using_unanimous_voting(self, min_confidence: float = None,
                                                  z_threshold=None) -> list:
        assert len(self.all_candidate_matches) > 0, "No candidate matches!"
        z_threshold, min_confidence = self.use_defaults_if_none(min_confidence, z_threshold)

        candidates = self.all_candidate_matches
        candidates = [c for c in candidates if c[2] > min_confidence]
        candidates = self.filter_matches_using_z_threshold(candidates, z_threshold)

        match_dict = defaultdict(list)
        conf_dict = defaultdict(list)
        for c in candidates:
            if match_dict[c[0]]:
                # Evaluates false if None or empty
                if match_dict[c[0]] == c[1]:
                    # Was the same match
                    conf_dict[(c[0], c[1])].append(c[2])
                else:
                    # Must remove the match from both dictionaries
                    match_dict[c[0]] = None
                    conf_dict[(c[0], c[1])] = None
            elif match_dict[c[0]] is None:
                continue
            else:
                # Is empty (not yet matched)
                match_dict[c[0]] = c[1]
                conf_dict[(c[0], c[1])].append(c[2])

        # conf_dict = {k: np.tanh(v) for k, v in conf_dict.items()}
        final_matches = [[k0, k1, np.mean(v)] for (k0, k1), v in conf_dict.items() if v is not None]
        # Use bipartite matching to remove overmatching
        matches, conf, _ = calc_bipartite_from_candidates(final_matches)
        final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]

        self.final_matches = final_matches
        return final_matches

    def use_defaults_if_none(self, min_confidence, z_threshold):
        if min_confidence is None:
            min_confidence = self.options.min_confidence
        else:
            self.options.min_confidence = min_confidence
        if z_threshold is None:
            z_threshold = self.options.z_threshold
        else:
            self.options.z_threshold = z_threshold
        return z_threshold, min_confidence

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

    def get_metadata_dict(self) -> FramePairOptions:
        return self.options

    def save_matches_as_excel(self, target_dir='.'):
        f0_ind = self.frame0.frame_ind
        f1_ind = self.frame1.frame_ind
        df = pd.DataFrame(self.final_matches, columns=[f'Neuron_in_f{f0_ind}', f'Neuron_in_f{f1_ind}', 'Confidence'])
        fname = f'matches_f{f0_ind}_f{f1_ind}.xlsx'
        fname = os.path.join(target_dir, fname)
        df.to_excel(fname, index=False)

    def filter_matches_using_z_threshold(self, matches, z_threshold) -> list:
        if z_threshold is None:
            return matches
        n0 = self.frame0.neuron_locs.copy()
        n1 = self.frame1.neuron_locs.copy()

        def _delta_z(m):
            return np.abs(n0[m[0]][0] - n1[m[1]][0])

        return [m for m in matches if _delta_z(m) < z_threshold]

    def modify_confidences_using_image_features(self, metadata: DetectedNeurons, gamma=1.0, mode='brightness'):
        # Get brightness... this object doesn't know the object, because it is full-video information
        i0, i1 = self.frame0.frame_ind, self.frame1.frame_ind
        if mode == 'brightness':
            x0, x1 = metadata.get_normalized_intensity(i0), metadata.get_normalized_intensity(i1)
        elif mode == 'volume':
            x0, x1 = metadata.get_all_volumes(i0), metadata.get_all_volumes(i1)
        else:
            raise ValueError

        # Per match, calculate similarity score based on delta
        matches = self.final_matches
        distances = [x0[m[0]] - x1[m[1]] for m in matches]
        multipliers = dist2conf(distances, gamma)

        # Multiply the original confidence
        matches[:, 2] *= multipliers

        return matches

    def print_candidates_by_method(self):
        num_matches = len(self.feature_matches)
        print(f"Found {num_matches} candidates via feature matching")
        num_matches = len(self.affine_matches)
        print(f"Found {num_matches} candidates via affine matching")
        num_matches = len(self.gp_matches)
        print(f"Found {num_matches} candidates via gaussian process matching")

        num_matches = len(self.final_matches)
        print(f"Processed these into {num_matches} final matches candidates")

    def print_candidates_for_neuron(self, i_neuron, i_frame=0):
        for m in self.all_candidate_matches:
            if m[i_frame] == i_neuron:
                print(f"Candidate: {m}")

    def print_reason_for_match(self, test_match):
        m0, m1 = test_match

        all_match_types = [
            (self.feature_matches, "feature"),
            (self.affine_matches, "affine"),
            (self.gp_matches, "gaussian process"),
            (self.fdnc_matches, "fdnc (neural network)"),
        ]

        f0_to_1 = self.get_f0_to_f1_dict()
        if f0_to_1[m0] == m1:
            f_to_conf = self.get_pair_to_conf_dict()
            print(f"Found match {test_match} with confidence {f_to_conf[test_match]}")

            for match_type in all_match_types:
                self.print_match_by_method(match_type[0], m0, m1, match_type[1])

    def print_match_by_method(self, this_method_matches, m0, m1, method_name):
        aff_dict = self.get_f0_to_f1_dict(this_method_matches)
        if m0 in aff_dict:
            if aff_dict[m0] == m1:
                conf = self.get_pair_to_conf_dict(this_method_matches)[(m0, m1)]
                print(f"Same match from {method_name} method with confidence: {conf}")
            else:
                conf = self.get_pair_to_conf_dict(this_method_matches)[(m0, aff_dict[m0])]
                print(f"Different match from {method_name} method: {aff_dict[m0]} with confidence: {conf}")
        else:
            print(f"Neuron not matched using {method_name} method")

    def match_using_local_affine(self):
        if not self.options.add_affine_to_candidates:
            return
        else:
            obj = self.options
            opt = dict(start_plane=obj.start_plane,
                       num_features_per_plane=obj.num_features_per_plane,
                       matches_to_keep=obj.affine_matches_to_keep,
                       use_GMS=obj.affine_matches_to_keep,
                       min_matches=obj.min_matches,
                       allow_z_change=obj.allow_z_change,
                       num_candidates=obj.affine_num_candidates)
            self._match_using_local_affine(**opt)

    def _match_using_local_affine(self, start_plane,
                                  num_features_per_plane,
                                  matches_to_keep,
                                  use_GMS,
                                  min_matches,
                                  allow_z_change,
                                  num_candidates):
        # Generate keypoints and match per slice
        frame0, frame1 = self.frame0, self.frame1
        # TODO: should I reread the data here?
        dat0, dat1 = frame0.get_raw_data(), frame1.get_raw_data()
        # Transpose because opencv needs it
        dat0 = np.transpose(dat0, axes=(0, 2, 1))
        dat1 = np.transpose(dat1, axes=(0, 2, 1))
        opt = dict(start_plane=start_plane,
                   num_features_per_plane=num_features_per_plane,
                   matches_to_keep=matches_to_keep,
                   use_GMS=use_GMS)
        kp0_locs, kp1_locs, all_kp0, all_kp1, kp_matches, all_match_offsets = \
            build_features_and_match_2volumes(dat0, dat1, **opt)
        # Save intermediate data in objects
        frame0.keypoint_locs = kp0_locs
        frame1.keypoint_locs = kp1_locs
        frame0.keypoints = all_kp0
        frame1.keypoints = all_kp1
        # kp_matches = recursive_cast_matches_as_array(kp_matches, all_match_offsets, gamma=1.0)
        kp_matches = [(i, i, 1.0) for i in range(len(kp0_locs))]
        self.keypoint_matches = kp_matches
        # Then match using distance from neuron position to keypoint cloud
        options = {'all_feature_matches': kp_matches,
                   'min_matches': min_matches,
                   'allow_z_change': allow_z_change,
                   'num_candidates': num_candidates}
        affine_matches, _, affine_pushed = calc_matches_using_affine_propagation(frame0, frame1, **options)
        # TODO: above code requires that the keypoint_locs are actually the full keypoints...
        # frame0.keypoint_locs = kp0_locs
        # frame1.keypoint_locs = kp1_locs
        self.affine_matches = affine_matches
        self.affine_pushed_locations = affine_pushed

    def match_using_gp(self):
        if not self.options.add_gp_to_candidates:
            return
        else:
            self._match_using_gp(self.options.gp_num_candidates, self.options.starting_matches)

    def _match_using_gp(self, n_neighbors, starting_matches_name='best'):
        if starting_matches_name == 'best':
            if len(self.affine_matches) > len(self.feature_matches):
                starting_matches_name = 'affine_matches'
            else:
                starting_matches_name = 'feature_matches'
        if starting_matches_name in ['affine_matches', 'feature_matches']:
            starting_matches = getattr(self, starting_matches_name)
        else:
            raise ValueError(f"Unknown starting matches: {starting_matches_name}")

        # Can start with any matched point clouds, but not more than ~100 matches otherwise it's way too slow
        frame0, frame1 = self.frame0, self.frame1
        n0 = frame0.neuron_locs.copy()
        n1 = frame1.neuron_locs.copy()
        n0[:, 0] *= self.options.z_to_xy_ratio
        n1[:, 0] *= self.options.z_to_xy_ratio
        # Actually match
        options = {'matches_with_conf': starting_matches, 'n_neighbors': n_neighbors}
        gp_matches, all_gps, gp_pushed = calc_matches_using_gaussian_process(n0, n1, **options)
        # gp_matches = recursive_cast_matches_as_array(gp_matches, gamma=1.0)
        self.gp_matches = gp_matches
        self.all_gps = all_gps
        self.gp_pushed_locations = gp_pushed

    def match_using_feature_embedding(self):
        # Default method; always call this
        obj = self.options
        opt = dict(matches_to_keep=obj.embedding_matches_to_keep,
                   use_GMS=obj.embedding_use_GMS,
                   crossCheck=obj.crossCheck)
        self._match_using_feature_embedding(**opt)

    def _match_using_feature_embedding(self, matches_to_keep=1.0, use_GMS=False, crossCheck=True):
        """
        Requires the frame objects to have been correctly initialized, i.e. their neurons need a feature embedding
        """
        frame0, frame1 = self.frame0, self.frame1
        # First, get feature matches
        neuron_embedding_matches = match_known_features(frame0.all_features,
                                                        frame1.all_features,
                                                        frame0.neuron_locs,
                                                        frame1.neuron_locs,
                                                        frame0.vol_shape[1:],
                                                        frame1.vol_shape[1:],
                                                        matches_to_keep=matches_to_keep,
                                                        use_GMS=use_GMS,
                                                        crossCheck=crossCheck)
        # With neuron embeddings, the keypoints are the neurons
        neuron_embedding_matches_with_conf = cast_matches_as_array(neuron_embedding_matches, gamma=1.0)
        self.feature_matches = neuron_embedding_matches_with_conf
        self.keypoint_matches = neuron_embedding_matches_with_conf  # Overwritten by affine match, if used

    def match_using_fdnc(self):
        if not self.options.add_fdnc_to_candidates:
            return
        else:
            self._match_using_fdnc(self.options.fdnc_options)

    def _match_using_fdnc(self, prediction_options):
        from fDNC.src.DNC_predict import predict_matches
        frame0, frame1 = self.frame0, self.frame1
        template_pos = zimmer2leifer(np.array(frame0.neuron_locs))
        test_pos = zimmer2leifer(np.array(frame1.neuron_locs))

        _, matches_with_conf = predict_matches(test_pos=test_pos, template_pos=template_pos, **prediction_options)
        if prediction_options['topn'] is not None:
            matches_with_conf = flatten_nested_list(matches_with_conf)
        self.fdnc_matches = matches_with_conf

    def print_reason_for_all_final_matches(self):
        dict_of_matches = self.get_pair_to_conf_dict()
        for k in dict_of_matches.keys():
            print("==================================")
            self.print_reason_for_match(k)

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)}/{self.num_possible_matches} matches \n"


def calc_FramePair_from_Frames(frame0: ReferenceFrame, frame1: ReferenceFrame, frame_pair_options: FramePairOptions,
                               verbose: int = 1,
                               DEBUG: bool = False) -> FramePair:
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    Main constructor for the class FramePair

    See also: calc_2frame_matches
    """
    # Frames are 'desynced' because affine matching overwrites the keypoints
    # frame0.check_data_desyncing()
    # frame1.check_data_desyncing()

    # Create class, then call member functions
    frame_pair = FramePair(options=frame_pair_options, frame0=frame0, frame1=frame1)
    # frame_pair = FramePair(frame0=frame0, frame1=frame1,
    #                        add_affine_to_candidates=add_affine_to_candidates,
    #                        add_gp_to_candidates=add_gp_to_candidates,
    #                        min_confidence=min_confidence)

    # Core matching algorithm
    frame_pair.match_using_feature_embedding()

    # Add additional candidates; the class checks if they are used
    frame_pair.match_using_local_affine()
    frame_pair.match_using_gp()
    frame_pair.match_using_fdnc()

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

    new_pair = calc_FramePair_from_Frames(frame0, frame1, metadata)
    new_pair.calc_final_matches_using_bipartite_matching(pair.options.min_confidence)

    return new_pair
