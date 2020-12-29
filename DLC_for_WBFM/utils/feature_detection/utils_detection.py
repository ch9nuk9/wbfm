import cv2
import numpy as np
import open3d as o3d


def detect_blobs(im1_raw):
    """
    Detects neuron-like blobs in a 2d image
    """
    im1 = cv2.GaussianBlur(im1_raw,(5,5),0)
    # im1 = cv2.bilateralFilter(im1_raw, 5, 0, 3)

    im1 = cv2.bitwise_not(im1)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = np.max(im1)
    params.thresholdStep = 1

    params.minDistBetweenBlobs = 2

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 25

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    #detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im1)

    return keypoints, im1


def build_point_clouds_for_volume(dat,
                                  num_slices,
                                  alpha):
    """
    Build point clouds for each plane, with points = neurons
    """

    all_keypoints_pcs = []
    all_ims_with_kps = []

    f = lambda dat, which_slice : (alpha*dat[which_slice]).astype('uint8')

    for i in range(num_slices):
        im1_raw = f(dat, i)
        kp, im1 = detect_blobs(im1_raw)
        # Add to make the format: ZXY
        kp_3d = np.array([np.hstack((i, row.pt)) for row in kp])

        pc = o3d.geometry.PointCloud()
        if len(kp_3d) > 0:
            pc.points = o3d.utility.Vector3dVector(kp_3d)

        all_keypoints_pcs.append(pc)

    return all_keypoints_pcs


def build_correspondence_icp(all_keypoints_pcs,
                            opt = {'max_correspondence_distance':4.0}):
    # Build correspondence between each pair of planes

    all_icp = []

    for i in range(len(all_keypoints_pcs)-1):
    #     print(f"{i} / {num_slices}")
        this_pc = all_keypoints_pcs[i]
        next_pc = all_keypoints_pcs[i+1]

        reg = o3d.pipelines.registration.registration_icp(this_pc, next_pc, **opt)

        all_icp.append(reg)

    return all_icp
