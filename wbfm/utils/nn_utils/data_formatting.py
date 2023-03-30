import numpy as np
from sklearn.preprocessing import StandardScaler

zimmer_fluroscence_um_per_pixel_xy = 0.325
zimmer_behavior_um_per_pixel_xy = 2.4
ZIMMER_UM_PER_PIXEL_Z = 1

leifer_um_per_unit = 84


def zimmer2physical_fluorescence(vol0_zxy: np.ndarray,
                                 zimmer_um_per_pixel_z: float = None) -> np.ndarray:
    if zimmer_um_per_pixel_z is None:
        zimmer_um_per_pixel_z = zimmer_um_per_pixel_z
    # xy, then z
    xy_in_um = vol0_zxy[:, [1, 2]] * zimmer_fluroscence_um_per_pixel_xy
    xy_in_physical = xy_in_um

    z_in_um = vol0_zxy[:, [0]] * zimmer_um_per_pixel_z
    z_in_physical = z_in_um

    zxy_in_phyical = np.hstack([z_in_physical, xy_in_physical])

    return zxy_in_phyical


def zimmer2physical_behavior(frame0_xy: np.ndarray) -> np.ndarray:
    # Assume no z coordinate
    xy_in_um = frame0_xy * zimmer_behavior_um_per_pixel_xy

    return xy_in_um


def zimmer2leifer(vol0_zxy: np.ndarray) -> np.ndarray:
    """ Target: 1 unit = 84 um"""

    # xy, then z
    xy_in_um = vol0_zxy[:, [1, 2]] * zimmer_fluroscence_um_per_pixel_xy
    xy_in_leifer = xy_in_um / leifer_um_per_unit

    z_in_um = vol0_zxy[:, [0]] * ZIMMER_UM_PER_PIXEL_Z
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


def flatten_nested_list(nested_list):
    nested_list = [item for sublist in nested_list for item in sublist]
    return nested_list