from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_alignment.class_reference_frames import PreprocessingSettings
from DLC_for_WBFM.utils.feature_alignment.utils_reference_frames import perform_preprocessing

import numpy as np
import os


## Get the data
dat_folder = ''
fname = 'test_volume.tif'
fname = os.path.join(dat_folder, fname)

import_opt = {'which_vol':0, 'num_slices':33}
dat_raw = get_single_volume(fname, **import_opt)

# Get rid of flyback
dat_raw = dat_raw[1:]

## Preprocessing

# Initilize settings
preprocessing_settings = PreprocessingSettings()
preprocessing_settings.do_filtering = False # Can take a LONG time
preprocessing_settings.do_rigid_alignment = True
# IF the data is initially uint16, this should be changed
preprocessing_settings.alpha = 1.0

# Actually do it... can take a while
dat = perform_preprocessing(dat_raw, preprocessing_settings)


## Look at the data
