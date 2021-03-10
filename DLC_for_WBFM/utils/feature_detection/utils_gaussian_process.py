import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn import preprocessing
from DLC_for_WBFM.utils.feature_detection.utils_features import build_neuron_tree
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance
import open3d as o3d


def calc_matches_using_gaussian_process(n0_unmatched, n1_unmatched,
                                        this_match,
                                        this_conf,
                                        max_dist=30.0):
    """
    Using noisy matches between 3d point clouds (format: zxy) does the following:
        1. Interpolate the vector field using Gaussian Processes
        2. Push all points in point cloud n0
        3. Find the closest matches using ICP
    """

    # Build regression vectors and z-score
    xyz = np.zeros((len(this_match), 3), dtype=np.float32) # Start point
    dat = np.zeros((len(this_match), 3), dtype=np.float32) # Difference vector
    noise = np.zeros(len(this_match), dtype=np.float32) # Heuristic noise
    for m, (match, conf) in enumerate(zip(this_match, this_conf)):
        v0 = n0_unmatched[match[0]]
        v1 = n1_unmatched[match[1]]
        xyz[m, :] = v0
        dat[m, :] = v1 - v0
        noise[m] = np.exp((1-conf)/1e-1) + 1e-10 # Maximum confidence should be 1.0
    noise /= 1e4*np.max(noise)

    scaler = preprocessing.StandardScaler().fit(xyz)
    xyz_scaled = scaler.transform(xyz)
    xyz_unmatched_scaled = scaler.transform(n0_unmatched)

    scaler2 = preprocessing.StandardScaler().fit(dat)
    dat_scaled = scaler2.transform(dat)

    # Fit 3 GPs for x, y, and z
    # Do each coordinate independently
    kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3,100)) + \
        RBF(length_scale=0.5, length_scale_bounds=(1e-08, 10.0))

    opt = {'n_restarts_optimizer':10, 'alpha':noise}
    gpx = GaussianProcessRegressor(kernel=kernel,**opt)
    gpx.fit(xyz_scaled[:,1:], dat_scaled[:,1])
    gpy = GaussianProcessRegressor(kernel=kernel,**opt)
    gpy.fit(xyz_scaled[:,1:], dat_scaled[:,2])
    gpz = GaussianProcessRegressor(kernel=kernel,**opt)
    gpz.fit(xyz_scaled[:,1:], dat_scaled[:,0])

    x_predict = gpx.predict(xyz_unmatched_scaled[:,1:])
    y_predict = gpy.predict(xyz_unmatched_scaled[:,1:])
    z_predict = gpz.predict(xyz_unmatched_scaled[:,1:])

    # Get back to original space
    zxy_predict = np.vstack([z_predict,x_predict,y_predict]).T
    zxy_predict = scaler2.inverse_transform(zxy_predict)

    # New: get matches using bipartite matching on distances
    # xyz0, xyz1 = n0_unmatched+zxy_predict, n1_unmatched
    # out = calc_bipartite_from_distance(xyz0, xyz1, max_dist=max_dist)
    # matches, conf, _ = out

    # Point cloud for the pushed and target neurons
    pc_pushed = build_neuron_tree(n0_unmatched+zxy_predict, False)[1]
    pc_target = build_neuron_tree(n1_unmatched, False)[1]

    # Get final matches using ICP
    opt = {'max_correspondence_distance':max_dist}
    reg = o3d.pipelines.registration.registration_icp(pc_pushed, pc_target, **opt)

    # Process, including a confidence value
    matches = np.asarray(reg.correspondence_set)
    conf_func = lambda x : 1 - (x/max_dist)
    conf = np.zeros((matches.shape[0],1))
    for i, (m0, m1) in enumerate(matches):
        dist = np.linalg.norm(n0_unmatched[m0] - n1_unmatched[m1])
        conf[i] = conf_func(dist)
    matches_with_conf = np.hstack([matches, conf])

    return matches_with_conf, pc_pushed, pc_target
