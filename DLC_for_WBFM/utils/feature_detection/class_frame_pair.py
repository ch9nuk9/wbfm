from dataclasses import dataclass
from DLC_for_WBFM.utils.feature_detection.utils_candidate_matches import calc_all_bipartite_matches

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

    # Original keypoints
    keypoint_matches: list = None

    @property
    def all_candidate_matches(self):
        tmp = self.feature_matches.copy()
        return tmp.extend(self.affine_matches).extend(self.gp_matches)

    def calc_all_bipartite_matches(self):
        return calc_all_bipartite_matches(self.all_candidate_matches)

    def calc_final_matches_using_bipartite_matching(self):
        self.final_matches = self.calc_all_bipartite_matches()