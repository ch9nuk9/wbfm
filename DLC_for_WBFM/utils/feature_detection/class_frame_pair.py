from dataclasses import dataclass
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

    all_gps: list = None  # The actual gaussian processes

    # Original keypoints
    keypoint_matches: list = None

    # Convinience
    # f0_to_f1_matches: dict = None
    # f1_to_f0_matches: dict = None

    # def __post_init__(self):
    #     self.f0_to_f1_matches =

    @property
    def all_candidate_matches(self):
        all_matches = self.feature_matches.copy()
        all_matches.extend(self.affine_matches)
        all_matches.extend(self.gp_matches)
        return all_matches

    def calc_final_matches_using_bipartite_matching(self) -> list:
        matches, conf, _ = calc_bipartite_from_candidates(self.all_candidate_matches)
        self.final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]
        return self.final_matches

    # def get_neuron_match(self):

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)} matches \n"
