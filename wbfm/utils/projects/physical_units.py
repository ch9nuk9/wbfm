import logging
from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicalUnitConversion:
    """Converts from pixels to micrometers, but also to Leifer-specific scaling"""

    zimmer_fluroscence_um_per_pixel_xy: float = 0.325
    zimmer_behavior_um_per_pixel_xy: float = 2.4
    zimmer_um_per_pixel_z: float = 1.5

    leifer_um_per_unit: float = 84

    volumes_per_second: float = None
    exposure_time: int = 12

    num_z_slices: int = None

    @property
    def frames_per_second(self):
        return self.volumes_per_second * self.num_z_slices

    @property
    def z_to_xy_ratio(self):
        return self.zimmer_um_per_pixel_z / self.zimmer_fluroscence_um_per_pixel_xy

    def zimmer2physical_fluorescence(self, vol0_zxy: np.ndarray) -> np.ndarray:
        """
        Assumes that z is the 0th dimension, and x/y are 1, 2

        Parameters
        ----------
        vol0_zxy - shape is N x 3, where N=number of neurons (or objects)

        Returns
        -------
        zxy_in_phyical - shape is same as input

        """
        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_physical = xy_in_um

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_physical = z_in_um

        zxy_in_phyical = np.hstack([z_in_physical, xy_in_physical])

        return zxy_in_phyical

    def zimmer2physical_fluorescence_single_column(self, dat0: np.ndarray, which_col=0) -> np.ndarray:
        """Converts just a single column, in place"""

        dat0[:, which_col] *= self.z_to_xy_ratio
        return dat0

    def zimmer2physical_behavior(self, frame0_xy: np.ndarray) -> np.ndarray:
        # Assume no z coordinate
        xy_in_um = frame0_xy * self.zimmer_behavior_um_per_pixel_xy

        return xy_in_um

    def zimmer2leifer(self, vol0_zxy: np.ndarray) -> np.ndarray:
        """ Target: 1 unit = 84 um, and xyz from zxy"""

        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_leifer = xy_in_um / self.leifer_um_per_unit

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_leifer = z_in_um / self.leifer_um_per_unit

        zxy_in_leifer = np.hstack([z_in_leifer, xy_in_leifer])
        xyz_in_leifer = zxy_in_leifer[:, [2, 1, 0]]

        xyz_in_leifer -= np.mean(xyz_in_leifer, axis=0)

        return xyz_in_leifer

    def leifer2zimmer(self, vol0_xyz_leifer: np.ndarray) -> np.ndarray:
        """Tries to invert zimmer2leifer, but does not know the original mean value"""

        # xy, then z
        xy_in_um = vol0_xyz_leifer[:, [0, 1]] * self.leifer_um_per_unit
        xy_in_zimmer = xy_in_um / self.zimmer_fluroscence_um_per_pixel_xy

        z_in_um = vol0_xyz_leifer[:, [2]] * self.leifer_um_per_unit
        z_in_zimmer = z_in_um / self.zimmer_um_per_pixel_z

        zxy_in_zimmer = np.hstack([z_in_zimmer, xy_in_zimmer])
        xyz_in_zimmer = zxy_in_zimmer[:, [2, 1, 0]]

        xyz_in_zimmer -= np.min(xyz_in_zimmer, axis=0)

        return xyz_in_zimmer

    @staticmethod
    def load_from_config(project_cfg):

        from wbfm.utils.general.postures.centerline_classes import get_behavior_fluorescence_fps_conversion
        if 'physical_units' in project_cfg.config:
            # Main units
            opt = project_cfg.config['physical_units']
            if 'volumes_per_second' not in opt:
                project_cfg.logger.debug("Using hard coded camera fps; this depends on the exposure time")
                camera_fps = opt.get('1000', 1000)
                exposure_time = opt.get('exposure_time', 12)
                frames_per_volume = get_behavior_fluorescence_fps_conversion(project_cfg)
                opt['volumes_per_second'] = camera_fps / exposure_time / frames_per_volume
            # Additional dataset unit
            opt_dataset = project_cfg.config['dataset_params']
            if 'num_slices' in opt_dataset:
                opt['num_z_slices'] = opt_dataset['num_slices']
            else:
                # This is a very old parameter, and should be in all projects
                raise ValueError("num_slices not found in dataset_params")

            return PhysicalUnitConversion(**opt)
        else:
            project_cfg.logger.warning("Using default physical unit conversions")
            return PhysicalUnitConversion()
