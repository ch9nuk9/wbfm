from dataclasses import dataclass


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