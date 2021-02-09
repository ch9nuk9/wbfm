from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import perform_preprocessing

import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile as tiff

def rigid_prealignment(fname):
    """
    Pre-aligns neurons on slices across Z.
    May cause large artefacts at the edges.

    Parameters
    ----------
    vol_path : str
        Path to volume, which shall be aligned
    Returns
    -------
    3D array with pre-aligned neurons
    """

    import_opt = {'which_vol':0, 'num_slices':33, 'alpha': 1.0, 'dtype': 'uint16'}
    dat_raw = get_single_volume(fname, **import_opt)

    # Get rid of flyback
    dat_raw = dat_raw[1:]

    ## Preprocessing

    # Initilize settings
    preprocessing_settings = PreprocessingSettings()
    preprocessing_settings.do_filtering = True  # Can take a LONG time
    preprocessing_settings.do_rigid_alignment = True
    # IF the data is initially uint16, this should be changed
    preprocessing_settings.alpha = 1.0
    preprocessing_settings.final_dtype = 'uint16'

    # Actually do it... can take a while
    dat = perform_preprocessing(dat_raw, preprocessing_settings)

    # optional plotting
    # Look at the data
#     plt.figure()
#     plt.imshow(np.max(dat, axis=0))
#     plt.title('maxP dat')
#
#     plt.figure()
#     plt.imshow(np.max(dat_raw, axis=0))
#     plt.title('maxP raw')
#     save preprocessed data as a new test volume
#     tiff.imsave(r'C:\Segmentation_working_area\test_volume\preprocessed_volume.tif', dat)
#
#     plt.show()

    return dat