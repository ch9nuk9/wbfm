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


# def encode_all_neurons_in_frame(frame, volume, z_depth):
#     """
#     Convinience wrapper around encode_all_neurons for a ReferenceFrame object
#
#     Also saves the output in the correct fields
#
#     Note that volume should be passed to ensure identical preprocessing
#     """
#
#     embeddings = encode_all_neurons(frame.neuron_locs, volume, z_depth)
#
#     frame.all_features = embeddings
#     frame.keypoint_locs = frame.neuron_locs
#     # This is now just a trivial mapping
#     frame.features_to_neurons = {i: i for i in range(len(frame.neurons_locs))}
#
#     return frame


def cast_matches_as_array(matches: list, gamma: object = 1.0) -> list:
    """
    Reformats a cv2 match object to a list of 3-element matches with a confidence score

    confidence = sigma(gamma / match.distance)
    """

    def conf(dist):
        return np.tanh(gamma / (dist + 1e-9 * gamma))

    return [(m.queryIdx, m.trainIdx, conf(m.distance)) for m in matches]
