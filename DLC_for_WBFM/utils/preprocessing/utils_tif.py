import numpy as np
import tifffile

from DLC_for_WBFM.utils.feature_detection.utils_rigid_alignment import align_stack, filter_stack
import scipy.ndimage as ndi
from dataclasses import dataclass
from dataclasses import field
from typing import List
import yaml
from tqdm import tqdm


##
## Class to hold preprocessing settings
##
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume


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

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, 'r') as f:
            cfg = yaml.safe_load(f)
        return PreprocessingSettings(**cfg)


def perform_preprocessing(dat_raw, preprocessing_settings: PreprocessingSettings):
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings
    if s is None:
        return dat_raw

    if s.do_filtering:
        dat_raw = filter_stack(dat_raw, s.filter_opt)

    if s.do_rigid_alignment:
        dat_raw = align_stack(dat_raw)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        dat_raw = ndi.maximum_filter(dat_raw, size=(mini_max_size,1,1))

    dat_raw = (dat_raw*s.alpha).astype(s.final_dtype)

    return dat_raw


def _preprocess_all_frames(DEBUG, config, verbose, vid_fname, which_frames=None):
    """
    Preproceses all frames that will be analyzed as per config

    Loads but does not process frames before config['dataset_params']['start_volume']
        (to keep the indices the same as the original dataset)
    """
    sz, vid_opt = _get_video_options(config, vid_fname)
    if verbose >= 1:
        print("Preprocessing data, this could take a while...")
    p = PreprocessingSettings.load_from_yaml(config['preprocessing_config'])
    start_volume = config['dataset_params']['start_volume']
    num_total_frames = start_volume + config['dataset_params']['num_frames']
    num_slices = config['dataset_params']['num_slices']
    if DEBUG:
        # Make a much shorter video
        num_total_frames = which_frames[-1] + 1
    chunk_sz = (num_slices, ) + sz
    total_sz = (num_total_frames, ) + chunk_sz
    preprocessed_dat = np.zeros(total_sz, dtype='uint16')
    # preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype='uint16',
    #                               synchronizer=zarr.ThreadSynchronizer())
    # read_lock = threading.Lock()
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    with tifffile.TiffFile(vid_fname) as vid_stream:
        # def parallel_func(i):
        #     preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream, read_lock)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=len(frame_list)) as executor:
        #     futures = executor.map(parallel_func, frame_list)
        #     [f.result() for f in futures]
        for i in tqdm(frame_list):
            preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream)
    return preprocessed_dat, vid_opt


def _get_video_options(config, vid_fname):
    with tifffile.TiffFile(vid_fname) as tif:
        sz = tif.pages[0].shape
    vid_opt = {'fps': config['dataset_params']['fps'],
               'frame_height': sz[0],
               'frame_width': sz[1]}
    return sz, vid_opt


def _get_and_preprocess(i, num_slices, p, start_volume, vid_fname, read_lock=None):
    if read_lock is None:
        dat_raw = get_single_volume(vid_fname, i, num_slices, dtype='uint16')
    else:
        with read_lock:
            dat_raw = get_single_volume(vid_fname, i, num_slices, dtype='uint16')
    # Don't preprocess data that we didn't even segment!
    if i >= start_volume:
        # preprocessed_dat[i, ...] = perform_preprocessing(dat_raw, p)
        return perform_preprocessing(dat_raw, p)
    else:
        # preprocessed_dat[i, ...] = dat_raw
        return dat_raw