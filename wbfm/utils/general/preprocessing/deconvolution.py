from dataclasses import dataclass
from typing import Optional

import numpy as np
import psf
from backports.cached_property import cached_property
from skimage import restoration


@dataclass
class DeconvolutionScaler:
    """Preprocesses data for deconvolution"""

    background_per_pixel: int

    max_val: Optional[int] = None
    initial_dtype: np.dtype = np.uint16

    def scale_volume(self, vol):
        if self.max_val is None:
            self.max_val = np.max(vol)
        return (vol - self.background_per_pixel) / self.max_val

    def unscale_volume(self, vol_scaled):
        return (vol_scaled * self.max_val) + self.background_per_pixel

    def reset(self):
        self.max_val = None


class CustomPSF:

    scaler: DeconvolutionScaler = None

    @cached_property
    def psf(self):
        # args = {
        #         'shape': (22, 650),  # number of samples in z and r direction
        #         'dims': (33, 211,25),  # size of FULL IMAGE in z and r direction in micrometers
        #         'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        #         'em_wavelen': 520.0,  # emission wavelength in nanometers
        #         'num_aperture': 1.0, #1.2,
        #         'refr_index': 1.333,
        #         'magnification': 40.0,
        #         'pinhole_radius': 1.25, #0.025,  # in micrometers
        #         'pinhole_shape': 'round',
        #     }

        # TODO: the shape affects this quite a lot
        args = {
            'shape': (7, 9),  # number of samples in z and r direction
            'dims': (10.5, 2.925),  # size of FULL IMAGE in z and r direction in micrometers
            'ex_wavelen': 488.0,  # excitation wavelength in nanometers
            'em_wavelen': 520.0,  # emission wavelength in nanometers
            'num_aperture': 1.0,
            'refr_index': 1.333,
            'magnification': 40.0,
            'pinhole_radius': 1.25, # in micrometers
            'pinhole_shape': 'round',
        }

        obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)

        return obsvol

    @cached_property
    def psf_2d(self):
        # TODO: is this the middle slice?
        obsvol = self.psf
        return obsvol.volume()[obsvol.shape[0], ...]

    def deconvolve_volume_and_save(self, i, input_zarr, output_zarr):
        vol = input_zarr[i]
        vol_deconvolved = self.deconvolve_single_volume_2d(vol)
        output_zarr[i] = vol_deconvolved

    def deconvolve_single_volume_2d(self, vol):
        """
        Deconvolves a volume using a 2d point spread function

        Parameters
        ----------
        vol

        Returns
        -------

        """
        if self.scaler is not None:
            vol = self.scaler.scale_volume(vol)
        psf_2d = self.psf_2d
        vol_deconvolved = np.zeros(vol.shape)
        for _i in range(vol.shape[0]):
            img = vol[_i, ...]
            out = restoration.richardson_lucy(img, psf_2d, iterations=30, filter_epsilon=1e-5)
            vol_deconvolved[_i, :, :] = out
        if self.scaler is not None:
            vol_deconvolved = self.scaler.unscale_volume(vol_deconvolved)
        return vol_deconvolved
