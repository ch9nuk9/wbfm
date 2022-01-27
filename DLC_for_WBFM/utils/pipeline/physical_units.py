from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicalUnitConversion:
    """Converts from pixels to micrometers, but also to Leifer-specific scaling"""

    zimmer_fluroscence_um_per_pixel_xy: float = 0.325
    zimmer_behavior_um_per_pixel_xy: float = 2.4
    zimmer_um_per_pixel_z: float = 1.0

    leifer_um_per_unit: float = 84

    def zimmer2physical_fluorescence(self, vol0_zxy: np.ndarray) -> np.ndarray:
        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_physical = xy_in_um

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_physical = z_in_um

        zxy_in_phyical = np.hstack([z_in_physical, xy_in_physical])

        return zxy_in_phyical

    def zimmer2physical_behavior(self, frame0_xy: np.ndarray) -> np.ndarray:
        # Assume no z coordinate
        xy_in_um = frame0_xy * self.zimmer_behavior_um_per_pixel_xy

        return xy_in_um

    def zimmer2leifer(self, vol0_zxy: np.ndarray) -> np.ndarray:
        """ Target: 1 unit = 84 um"""

        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_leifer = xy_in_um / self.leifer_um_per_unit

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_leifer = z_in_um / self.leifer_um_per_unit

        zxy_in_leifer = np.hstack([z_in_leifer, xy_in_leifer])
        xyz_in_leifer = zxy_in_leifer[:, [2, 1, 0]]

        xyz_in_leifer -= np.mean(xyz_in_leifer, axis=0)

        return xyz_in_leifer
