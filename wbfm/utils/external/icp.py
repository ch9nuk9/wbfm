import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def icp(a, b, init_pose=(0, 0, 0), no_iterations=13, verbose=0):
    '''
    FROM: https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    # Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                   [0, 0, 1]])

    # src = cv2.transform(src, Tr[0:2])
    src = cv2.transform(src, Tr)
    res = np.inf
    all_res = []

    for i in range(no_iterations):
        # Find the nearest neighbours between the current source and the
        # destination cloudpoint
        # i.e. impose an ordering
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])
        res = np.linalg.norm(distances)
        all_res.append(res)
        if verbose >= 2:
            print(f"residual at step {i}: {res}")

        # Compute the transformation between the current source
        # and destination cloudpoint
        # T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        # val, h, inliers = cv2.estimateAffine3D(src, dst[0, indices.T])
        h, inliers = cv2.estimateAffine2D(src[:, :, 1:], dst[0, indices.T][:, :, 1:], confidence=0.9999)
        if h is not None:
            # Transform the previous source and update the
            # current source cloudpoint
            # src = cv2.transform(src, h)
            src[:, :, 1:] = cv2.transform(src, h)
            # Save the transformation from the actual source cloudpoint
            # to the destination
            Tr = np.dot(Tr, np.vstack((h, [0, 0, 1])))
        else:
            print(f"No valid transformation; quitting at iteration {i}")
            break

    return Tr[0:2], all_res
