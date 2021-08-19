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

    all_gps: list = None  # The actual gaussian processes; may get warning from scipy versioning

    # Original keypoints
    keypoint_matches: list = None

    min_confidence: float = 0.001

    @property
    def all_candidate_matches(self):
        all_matches = self.feature_matches.copy()
        all_matches.extend(self.affine_matches)
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

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)} matches \n"
