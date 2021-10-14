import numpy as np
from sklearn.preprocessing import StandardScaler

zimmer_um_per_pixel = 0.325
zimmer_um_per_pixel_z = 1

leifer_um_per_unit = 84


def zimmer2leifer(vol0_zxy):
    """ Target: 1 unit = 84 um"""
    #
    # scaler = StandardScaler()
    # vol0_scaled = scaler.fit_transform(vol0_zxy)
    # # Reduce z
    # vol0_scaled[:, 0] /= 3.0
    # # Reorder dimensions
    # vol0_scaled = vol0_scaled[:, [2, 1, 0]]
    # # Somehow their point clouds are much smaller than mine
    # vol0_scaled /= 5.0

    # xy, then z
    xy_in_um = vol0_zxy[:, [1, 2]] * zimmer_um_per_pixel
    xy_in_leifer = xy_in_um / leifer_um_per_unit

    z_in_um = vol0_zxy[:, [0]] * zimmer_um_per_pixel_z
    z_in_leifer = z_in_um / leifer_um_per_unit

    zxy_in_leifer = np.hstack([z_in_leifer, xy_in_leifer])
    xyz_in_leifer = zxy_in_leifer[:, [2, 1, 0]]

    xyz_in_leifer -= np.mean(xyz_in_leifer, axis=0)

    return xyz_in_leifer


def leifer2zimmer(vol0_scaled, scaler):

    # Somehow their point clouds are much smaller than mine
    vol0_scaled *= 5.0
    # Reorder dimensions; coincidentally symmetric
    vol0_scaled = vol0_scaled[:, [2, 1, 0]]
    # Increase z
    vol0_scaled[:, 0] *= 3.0

    vol0_zxy = scaler.inverse_transform(vol0_scaled)

    return vol0_zxy
