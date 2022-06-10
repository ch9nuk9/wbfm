from typing import List

import cv2
import numpy as np


def get_keypoints_from_3dseg(kp0, i=None, sz=31.0, neuron_height=None):
    """Translate numpy array to cv2.Keypoints, based off one slice

    Parameters
    ----------
    kp0 : array-like
        Original 3d positions
    i : int
        current slice
        if None (default), adds all keypoints
    sz : float
        Size for cv2 keypoints... not sure if this matters
    neuron_height : float
        Radius around original annotations to add the keypoint to
        if None (default), adds all keypoints

    Returns
    -------
    type
        Description of returned object.

    """
    kp_cv2 = []
    for z, x, y in kp0:
        if i is None or abs(z - i) < neuron_height:
            kp_cv2.append(cv2.KeyPoint(y, x, sz))

    return kp_cv2


def recursive_cast_matches_as_array(list_of_cv2_matches: List[List], all_match_offsets: List, gamma: float):
    """
    Reformats a list of cv2 match objects (for different images, e.g. z slices) as one global list of matches

    Assumes the keypoints were concatenated, e.g. using .extend()

    Parameters
    ----------
    all_match_offsets
    list_of_cv2_matches
    gamma

    Returns
    -------

    """

    final_matches = []
    # index_offset = 0
    for cv2_matches, index_offset in zip(list_of_cv2_matches, all_match_offsets):
        matches_with_conf = cast_matches_as_array(cv2_matches, gamma, index_offset=index_offset)
        final_matches.append(matches_with_conf)
        # index_offset += len(matches_with_conf)

    return np.vstack(final_matches)


def cast_matches_as_array(cv2_matches: list, gamma: float = 1.0, index_offset=None) -> list:
    """
    Reformats a cv2 match object to a list of 3-element matches with a confidence score

    confidence = sigma(gamma / match.distance)
    """

    if index_offset is None:
        index_offset = [0, 0]

    def conf(dist):
        return np.tanh(gamma / (dist + 1e-9 * gamma))

    return [(m.queryIdx+index_offset[0], m.trainIdx+index_offset[1], conf(m.distance)) for m in cv2_matches]
