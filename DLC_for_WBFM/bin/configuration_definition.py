from dataclasses import dataclass


@dataclass
class DLC_for_WBFM_preprocessing:
    """
    Variables used for preprocessing
    """

    # bigtiff processing
    start_volume: int
    num_frames: int
    num_slices: int
    start_slice: int
    alpha: float

    # As of Nov 2020
    red_and_green_mirrored: bool = True


@dataclass
class DLC_for_WBFM_datafiles:
    """
    This project uses several very large of z-stack datafiles (~200 GB)

    This class collects all the original data filenames
        Also: created subfiles, e.g. 2d videos
    """

    # Original 3d files
    red_bigtiff_fname: str
    green_bigtiff_fname: str


@dataclass
class DLC_for_WBFM_tracking:
    """
    Collects input and output for a DLC run on a single slice
    """

    # One DLC run (input)
    original_video_fname: str
    DLC_config_fname: str
    DLC_project_foldername: str

    # One DLC run (output)
    labeled_video_fname: str
    annotation_fname: str


@dataclass
class DLC_for_WBFM_traces:
    """
    Collects files related to traces extracted AFTER tracking
    """

    # Parameters to current algorithm
    which_z: int # Center
    crop_sz: tuple
    # Note: also uses values from the preprocessing portion

    traces_fname: str


@dataclass
class DLC_for_WBFM_config:
    """
    Variables used for entire project workflow
    """
    # Overall project settings
    experimenter: str

    datafiles: DLC_for_WBFM_datafiles
    preprocessing: DLC_for_WBFM_preprocessing
    tracking: DLC_for_WBFM_single_slice
    traces: DLC_for_WBFM_traces
