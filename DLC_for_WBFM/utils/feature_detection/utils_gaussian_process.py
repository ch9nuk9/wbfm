import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn import preprocessing
from DLC_for_WBFM.utils.feature_detection.utils_features import build_neuron_tree
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance, calc_icp_matches
import open3d as o3d


def calc_matches_using_gaussian_process(n0_unmatched, n1_unmatched,
                                        matches_with_conf,
                                        max_dist=30.0):
    """
    Using noisy matches between 3d point clouds (format: zxy) does the following:
        1. Interpolate the vector field using Gaussian Processes
        2. Push all points in point cloud n0
        3. Find the closest matches using bipartite matching

    See also:
        calc_matches_using_feature_voting
        calc_matches_using_affine_propagation
    """

    if len(matches_with_conf) == 0:
        return [], (None, None, None), np.array([])
    # Build regression vectors and z-score
    xyz = np.zeros((len(matches_with_conf), 3), dtype=np.float32) # Start point
    dat = np.zeros((len(matches_with_conf), 3), dtype=np.float32) # Difference vector
    noise = np.zeros(len(matches_with_conf), dtype=np.float32) # Heuristic noise
    for m, (match_and_conf) in enumerate(matches_with_conf):
        v0 = n0_unmatched[match_and_conf[0]]
        v1 = n1_unmatched[match_and_conf[1]]
        xyz[m, :] = v0
        dat[m, :] = v1 - v0
        noise[m] = np.exp((1-match_and_conf[2])/1e-1) + 1e-10 # Maximum confidence should be 1.0
    noise /= 1e4*np.max(noise)

    scaler = preprocessing.StandardScaler().fit(xyz)
    xyz_scaled = scaler.transform(xyz)
    xyz_unmatched_scaled = scaler.transform(n0_unmatched)

    scaler2 = preprocessing.StandardScaler().fit(dat)
    dat_scaled = scaler2.transform(dat)

    # Fit 3 GPs for x, y, and z
    # Do each coordinate independently
    kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 100)) + \
        RBF(length_scale=0.5, length_scale_bounds=(1e-08, 10.0))

    opt = {'n_restarts_optimizer': 10, 'alpha': noise}
    gpx = GaussianProcessRegressor(kernel=kernel, **opt)
    gpx.fit(xyz_scaled[:, 1:], dat_scaled[:, 1])
    gpy = GaussianProcessRegressor(kernel=kernel, **opt)
    gpy.fit(xyz_scaled[:, 1:], dat_scaled[:, 2])
    gpz = GaussianProcessRegressor(kernel=kernel, **opt)
    gpz.fit(xyz_scaled[:, 1:], dat_scaled[:, 0])

    x_predict = gpx.predict(xyz_unmatched_scaled[:, 1:])
    y_predict = gpy.predict(xyz_unmatched_scaled[:, 1:])
    z_predict = gpz.predict(xyz_unmatched_scaled[:, 1:])

    # Get back to original space
    zxy_predict = np.vstack([z_predict, x_predict, y_predict]).T
    zxy_predict = scaler2.inverse_transform(zxy_predict)

    # Point cloud for the pushed and target neurons
    pc_pushed = build_neuron_tree(n0_unmatched+zxy_predict, False)[1]
    pc_target = build_neuron_tree(n1_unmatched, False)[1]

    # New: get matches using bipartite matching on distances
    xyz0, xyz1 = pc_pushed.points, pc_target.points
    # out = calc_bipartite_from_distance(xyz0, xyz1, max_dist=max_dist)
    out = calc_icp_matches(xyz0, xyz1, max_dist=max_dist)
    matches, conf, _ = out

    matches_with_conf = [(m[0], m[1], c[0]) for m, c in zip(matches, conf)]

    return matches_with_conf, (gpz, gpx, gpy), np.array(xyz0)
