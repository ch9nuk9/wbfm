from DLC_for_WBFM.utils.feature_detection.utils_features import convert_to_grayscale
import cv2
import numpy as np

from scipy.special import expit


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
    for z,x,y in kp0:
        if i is None or abs(z-i) < neuron_height:
            kp_cv2.append(cv2.KeyPoint(y,x,sz))

    return kp_cv2


def encode_all_neurons(locs_zxy, im_3d, z_depth):
    """
    Builds a feature vector for each neuron (zxy location) in a 3d volume
    Uses opencv VGG as a 2d encoder for a number of slices above and below the exact z location

    """
    im_3d_gray = [convert_to_grayscale(xy) for xy in im_3d]
    all_embeddings = []
    all_keypoints = []
    encoder = cv2.xfeatures2d.VGG_create()

    # Loop per neuron
    for loc in locs_zxy:
        z, x, y = loc
        kp = cv2.KeyPoint(x, y, 31.0)

        z = int(z)
        all_slices = np.arange(z - z_depth, z + z_depth + 1)
        all_slices = np.clip(all_slices, 0, len(im_3d_gray))
        # Generate features on neighboring z slices as well
        # Repeat slices if near the edge
        ds = []
        for i in all_slices:
            im_2d = im_3d_gray[int(i)]
            _, this_ds = encoder.compute(im_2d, kp)
            ds.append(this_ds)

        ds = np.hstack(ds)
        all_embeddings.extend(ds)
        all_keypoints.append(kp)

    return all_embeddings, all_keypoints


def encode_all_neurons_in_frame(frame, volume, z_depth):
    """
    Convinience wrapper around encode_all_neurons for a ReferenceFrame object

    Also saves the output in the correct fields

    Note that volume should be passed to ensure identical preprocessing
    """

    embeddings = encode_all_neurons(frame.neuron_locs, volume, z_depth)

    frame.all_features = embeddings
    frame.keypoint_locs = frame.neuron_locs
    # This is now just a trivial mapping
    frame.features_to_neurons = {i: i for i in range(len(frame.neurons_locs))}

    return frame


def match_object_to_array(matches, gamma=1.0):
    """
    Reformats a cv2 match object to a list of 3-element matches with a confidence score

    confidence = sigma(gamma / match.distance)
    """
    def conf(dist):
        return expit(gamma / dist)
    return [(m.queryIdx, m.trainIdx, conf(m.distance)) for m in matches]