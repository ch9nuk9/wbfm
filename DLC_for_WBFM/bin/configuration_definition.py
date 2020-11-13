from dataclasses import dataclass
import tifffile
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import get_crop_coords3d


@dataclass
class DLC_for_WBFM_preprocessing:
    """
    Variables used for preprocessing
    """

    # bigtiff processing
    # In time
    start_volume: int
    num_frames: int
    # In z
    num_total_slices: int
    num_crop_slices: int
    center_slice: int
    alpha: float # For conversion to uint8

    # As of Nov 2020
    red_and_green_mirrored: bool = True

    def which_slices(self):
        return list( get_crop_coords3d((0,0,self.center_slice),
                                (1,1,self.num_crop_slices) )[-1] )



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

    # Place to initially write the videos
    red_avi_fname: str
    green_avi_fname: str

    def get_frame_size(self):
        # Assume red and green are same size
        with tifffile.TiffFile(self.red_bigtiff_fname) as tif:
            frame_height, frame_width = tif.pages[0].shape

        return frame_height, frame_width


@dataclass
class DLC_for_WBFM_tracking:
    """
    Collects input and output for a DLC run on a single slice
    """

    # One DLC run (input)
    DLC_project_foldername: str

    # One DLC run (output)
    labeled_video_fname: str = None
    annotation_fname: str = None


@dataclass
class DLC_for_WBFM_traces:
    """
    Collects files related to traces extracted AFTER tracking
    """

    # Parameters to current algorithm
    is_3d: bool
    crop_sz: tuple
    # Note: also uses values from the preprocessing portion

    # TODO: which folder should these go in?
    traces_fname: str

    which_neurons: str = None


@dataclass
class DLC_for_WBFM_config:
    """
    Variables used for entire project workflow
    """
    # Overall project settings
    experimenter: str

    datafiles: DLC_for_WBFM_datafiles = None
    preprocessing: DLC_for_WBFM_preprocessing = None
    tracking: DLC_for_WBFM_tracking = None
    traces: DLC_for_WBFM_traces = None



def load_config(fname_or_config):
    """
    Helper to check if you passed the filename or the object itself
    """

    if isinstance(fname_or_config, str):
        return pickle.load(open(fname_or_config, 'rb'))
    else:
        assert isinstance(fname_or_config, DLC_for_WBFM_config), "Must be file path or DLC_for_WBFM_config"
        return fname_or_config
