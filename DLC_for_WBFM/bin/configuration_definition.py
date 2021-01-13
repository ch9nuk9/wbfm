from dataclasses import dataclass
import tifffile
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
from datetime import datetime as dt
import pickle
import os
import pathlib


@dataclass
class DLCForWBFMDatafiles:
    """
    This project uses several very large of z-stack datafiles (~200 GB)

    This class collects all the original data filenames
        Also: created subfiles, e.g. 2d videos
    """

    # Original 3d files
    red_bigtiff_fname: str = None
    green_bigtiff_fname: str = None

    # Place to initially write the videos
    red_avi_fname: str = None
    green_avi_fname: str = None

    def get_frame_size(self):
        # Assume red and green are same size
        with tifffile.TiffFile(self.red_bigtiff_fname) as tif:
            frame_height, frame_width = tif.pages[0].shape

        return frame_height, frame_width


@dataclass
class DLCForWBFMPreprocessing:
    """
    Variables used for preprocessing
    """

    # bigtiff processing
    # In time
    start_volume: int
    num_frames: int
    fps: float

    alpha: float # For conversion to uint8
    # In z
    num_total_slices: int = None
    num_crop_slices: int = None
    center_slice: int = None

    # As of Nov 2020
    red_and_green_mirrored: bool = True

    def which_slices(self):
        return list( get_crop_coords3d((0,0,self.center_slice),
                                (1,1,self.num_crop_slices) )[-1] )



@dataclass
class DLCForWBFMTracking:
    """
    Collects input and output for a DLC run on a single video
    """

    # One DLC config file
    # Future: list?
    DLC_config_fname: str

    # One DLC run (output)
    labeled_video_fname: str = None
    annotation_fname: str = None


@dataclass
class DLCForWBFMTraces:
    """
    Collects files related to traces extracted AFTER tracking
    """

    # Parameters to current algorithm
    is_3d: bool
    crop_sz: tuple
    # Note: also uses values from the preprocessing portion

    # Future: which folder should these go in?
    traces_fname: str

    which_neurons: str = None


@dataclass
class DLCForWBFMSegmentation:
    """
    Segmentation related variables

    WIP

    See also: cellpose
    """

    diameter: int = 8


@dataclass
class DLCForWBFMConfig:
    """Master configuration data for a DLC_for_WBFM project

    Parameters
    ----------
    task_name : str
        Descriptive string
    experimenter : str
        Experimenter name
    datafiles : DLCForWBFMDatafiles
        Pathnames for the raw data files (4d videos)
    preprocessing: DLCForWBFMPreprocessing
        Parameters for the preprocessing tasks, especially subslicing
    tracking: DLCForWBFMTracking
        Actual DeepLabCut settings for tracking
    traces: DLCForWBFMTraces
        Parameters for extracting traces, currently using dNMF
    config_filename: str
        Full path for this file; used for saving itself

    """
    # Overall project settings
    task_name: str
    experimenter: str

    datafiles: DLCForWBFMDatafiles = None
    preprocessing: DLCForWBFMPreprocessing = None
    tracking: DLCForWBFMTracking = None
    traces: DLCForWBFMTraces = None

    config_filename: str = None
    verbose: int = 1

    def get_dirname(self):
        return os.path.dirname(self.config_filename)

    def __str__(self):
        return f"=======================================\n\
                Field values:\n\
                task_name: {self.task_name} \n\
                experimenter: {self.experimenter} \n\
                config_filename: {self.config_filename}\n\
                =======================================\n\
                Which subclasses are initialized?\n\
                datafiles: {self.datafiles is not None}\n\
                preprocessing: {self.preprocessing is not None}\n\
                tracking: {self.tracking is not None}\n\
                traces: {self.traces is not None}\n"


def load_config(fname_or_config, always_reload=True):
    """
    Helper to check if you passed the filename or the object itself

    By default reloads the file from disk to make sure they are synchronized
    """

    if isinstance(fname_or_config, str):
        with open(fname_or_config, 'rb') as f:
            config = pickle.load(f)
    elif isinstance(fname_or_config, DLCForWBFMConfig):
        if always_reload:
            with open(fname_or_config.config_filename, 'rb') as f:
                config = pickle.load(f)
        else:
            return fname_or_config
    else:
        raise TypeError#, "Must be file path or DLCForWBFMConfig"

    return config



def save_config(config):
    """
    Saves config file in the location the object remembers
    i.e. config.config_filename

    # Future: do basic checks
    - Right operating system
    - Not a different project
    """

    fname = config.config_filename
    if '.pickle' not in fname:
        fname = fname + ".pickle"
    if config.verbose >= 2:
        print(f"Saving config to filename '{fname}''")
    with open(fname, 'wb') as f:
        pickle.dump(config, f)


def create_project(
    config,
    working_directory=None
):
    """
    Initializes a project given a parent folder and the config object
    Note: tries to make a short foldername (Windows might max out characters)

    Recommended workflow:
        Initialize a config object
        Create the file structure using this function
        Then, start preprocessing!

    Parameters
    ----------
    config : DLCForWBFMConfig
        Configuration object
    working_directory : str
        Path to parent folder

    Returns
    -------
    Nothing; creates folder and saves the config file there

    """

    config = load_config(config, always_reload=False)

    project_name = build_project_name(config)
    if working_directory == None:
        working_directory = "."
    wd = pathlib.Path(working_directory).resolve()
    project_path = wd / project_name

    # Create project and sub-directories
    if project_path.exists():
    # if not DEBUG and project_path.exists():
        if config.verbose >= 1:
            print('Project "{}" already exists!'.format(project_path))
    else:
        project_path.mkdir()
        if config.verbose >= 1:
            print(f'Created Project folder "{project_path}"')

    # Finally, save the config file in this folder
    config_filename = os.path.join(project_path,"config.pickle")
    config.config_filename = config_filename

    save_config(config)

    return config


##
## Filename builders
##

def build_project_name(config):
    """
    Build project name using the task_name, experimenter, and date
    """

    task_name, experimenter = config.task_name, config.experimenter

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    project_name = f"{task_name}-{experimenter}-{date}"

    return project_name


def build_avi_fnames(config):
    """
    Builds avi fnames if they don't exist
    Only replaces them if they are "None"

    Locates them in the parent folder, i.e. where the config file itself is
    """

    c = load_config(config)
    dir_name = c.get_dirname()

    frames = c.preprocessing.num_frames
    which_slices = c.preprocessing.which_slices()
    start, end = which_slices[0], which_slices[-1]

    suffix = f'fr{frames}_sl{start}_{end}.avi'

    if c.datafiles.green_avi_fname is None:
        green_avi_fname = os.path.join(dir_name, "green"+suffix)
        c.datafiles.green_avi_fname = green_avi_fname

    if c.datafiles.red_avi_fname is None:
        red_avi_fname = os.path.join(dir_name, "red"+suffix)
        c.datafiles.red_avi_fname = red_avi_fname

    save_config(c)

    return c
