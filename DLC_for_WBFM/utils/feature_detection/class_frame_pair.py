from dataclasses import dataclass
from DLC_for_WBFM.utils.feature_detection.utils_candidate_matches import calc_all_bipartite_matches

@dataclass
class FramePair:
    """Information connecting neurons in two ReferenceFrame objects"""

    # Final output
    final_matches: list
    final_conf: list

    # Intermediate products, with confidences
    feature_matches: list
    affine_matches: list
    gp_matches: list

    # Original keypoints
    keypoint_matches: list

    @property
    def all_candidate_matches(self):
        tmp = self.feature_matches.copy()
        return tmp.extend(self.affine_matches).extend(self.gp_matches)

    def calc_all_bipartite_matches(self):
        return calc_all_bipartite_matches(self.all_candidate_matches)