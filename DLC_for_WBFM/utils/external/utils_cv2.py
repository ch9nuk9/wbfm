
import cv2

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
