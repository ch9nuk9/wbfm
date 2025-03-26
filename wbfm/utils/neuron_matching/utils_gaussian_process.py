import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel
from sklearn.utils._testing import ignore_warnings
from tqdm.auto import tqdm

from wbfm.utils.general.utils_features import build_neuron_tree
from wbfm.utils.neuron_matching.utils_matching import calc_nearest_neighbor_matches


def calc_matches_using_gaussian_process(n0_unmatched, n1_unmatched,
                                        matches_with_conf,
                                        max_dist=30.0,
                                        n_neighbors=1):
    """
    Using noisy matches between 3d point clouds (format: zxy) does the following:
        1. Interpolate the vector field using Gaussian Processes
        2. Push all points in point cloud n0
        3. Find the closest matches using bipartite matching

    See also:
        calc_matches_using_feature_voting
        calc_matches_using_affine_propagation
    """

    if matches_with_conf is None or len(matches_with_conf) == 0:
        return [], (None, None, None), np.array([])
    # Build regression vectors and z-score
    xyz = np.zeros((len(matches_with_conf), 3), dtype=np.float32)  # Start point
    dat = np.zeros((len(matches_with_conf), 3), dtype=np.float32)  # Difference vector
    noise = np.zeros(len(matches_with_conf), dtype=np.float32)  # Heuristic noise
    for m, (match_and_conf) in enumerate(tqdm(matches_with_conf, leave=False)):
        v0 = n0_unmatched[match_and_conf[0]]
        v1 = n1_unmatched[match_and_conf[1]]
        xyz[m, :] = v0
        dat[m, :] = v1 - v0
        noise[m] = np.exp((1 - match_and_conf[2]) / 1e-1) + 1e-10  # Maximum confidence should be 1.0
    noise /= 1e4 * np.max(noise)

    scaler = preprocessing.StandardScaler().fit(xyz)
    xyz_scaled = scaler.transform(xyz)
    xyz_unmatched_scaled = scaler.transform(n0_unmatched)

    scaler2 = preprocessing.StandardScaler().fit(dat)
    dat_scaled = scaler2.transform(dat)

    # Fit 3 GPs for x, y, and z
    # Do each coordinate independently
    k0 = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 100))
    k1 = RBF(length_scale=0.5, length_scale_bounds=(1e-08, 10.0))
    kernel = k0 + k1

    options = {'n_restarts_optimizer': 10, 'alpha': noise}

    @ignore_warnings(category=ConvergenceWarning)
    def f():
        # See: https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        gpx = GaussianProcessRegressor(kernel=kernel, **options)
        gpx.fit(xyz_scaled[:, 1:], dat_scaled[:, 1])
        gpy = GaussianProcessRegressor(kernel=kernel, **options)
        gpy.fit(xyz_scaled[:, 1:], dat_scaled[:, 2])
        return gpx, gpy
    gpx, gpy = f()
    # gpz = GaussianProcessRegressor(kernel=kernel, **options)
    # gpz.fit(xyz_scaled[:, 1:], dat_scaled[:, 0])

    x_predict = gpx.predict(xyz_unmatched_scaled[:, 1:])
    y_predict = gpy.predict(xyz_unmatched_scaled[:, 1:])
    # z_predict = gpz.predict(xyz_unmatched_scaled[:, 1:])
    z_predict = xyz_unmatched_scaled[:, 0]  # DO NOT CHANGE Z

    # Get back to original space
    zxy_predict = np.vstack([z_predict, x_predict, y_predict]).T
    zxy_predict = scaler2.inverse_transform(zxy_predict)

    # Point cloud for the pushed and target neurons
    pc_pushed = build_neuron_tree(n0_unmatched + zxy_predict, False)[1]
    pc_target = build_neuron_tree(n1_unmatched, False)[1]

    # New: get matches using bipartite matching on distances
    xyz0, xyz1 = pc_pushed.points, pc_target.points
    matches, conf = calc_nearest_neighbor_matches(xyz0, xyz1, max_dist=max_dist, n_neighbors=n_neighbors)

    assert np.isscalar(conf[0]), "Check formatting (nested lists)"
    matches_with_conf = [(m[0], m[1], c) for m, c in zip(matches, conf)]

    return matches_with_conf, (gpx, gpy), np.array(xyz0)


def upsample_using_gaussian_process(y_raw, num_pts=50):
    """Designed to be used with brightness across z"""

    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * \
             RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

    scaler = preprocessing.StandardScaler()
    y = scaler.fit_transform(np.array(y_raw).reshape(-1, 1))
    max_z = y.shape[0]
    x = np.arange(max_z).reshape(-1, 1)

    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x, y)

    x2 = np.linspace(0, max_z - 1, num=num_pts).reshape(-1, 1)
    y2 = gp.predict(x2)
    y2_rescaled = np.squeeze(scaler.inverse_transform(y2))

    return y2_rescaled, gp, scaler
