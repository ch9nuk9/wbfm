from DLC_for_WBFM.utils.feature_detection.utils_rigid_alignment import align_stack, filter_stack
import scipy.ndimage as ndi
import numpy as np


##
## Class to hold preprocessing settings
##

@dataclass
class PreprocessingSettings():
    """
    Holds settings that will be applied to the ReferenceFrame class
    """

    # Filtering
    do_filtering : bool = False
    filter_opt : List = field(default_factory=lambda: {'high_freq':2.0, 'low_freq':5000.0})

    # Mini max
    do_mini_max_projection : bool = False
    mini_max_size : int = 3

    # Rigid alignment (slices to each other)
    do_rigid_alignment : bool = False

    # Datatypes and scaling
    initial_dtype : str = 'uint16' # Filtering etc. will act on this
    final_dtype : str = 'uint8'
    alpha : float = 0.15


def perform_preprocessing(dat_raw, preprocessing_settings:PreprocessingSettings):
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings

    if s.do_filtering:
        dat_raw = filter_stack(dat_raw, s.filter_opt)

    if s.do_rigid_alignment:
        dat_raw = align_stack(dat_raw)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        dat_raw = ndi.maximum_filter(dat_raw, size=(mini_max_size,1,1))

    dat_raw = (dat_raw*s.alpha).astype(s.final_dtype)

    return dat_raw
