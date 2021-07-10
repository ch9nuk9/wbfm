import concurrent
import dataclasses
import pickle
import threading
import typing

import numpy as np
import tifffile
import zarr

from DLC_for_WBFM.utils.feature_detection.utils_rigid_alignment import align_stack, filter_stack, \
    align_stack_using_previous_results
import scipy.ndimage as ndi
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple
import yaml
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.utils_project import edit_config
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume


##
## Class to hold preprocessing settings
##


@dataclass
class PreprocessingSettings:
    """
    Holds settings that will be applied to a video (.tiff or .btf)

    Designed to be used with the ReferenceFrame class
    """

    # Filtering
    do_filtering: bool = False
    filter_opt: List = field(default_factory=lambda: {'high_freq': 2.0, 'low_freq': 5000.0})

    # Mini max
    do_mini_max_projection: bool = False
    mini_max_size: int = 3

    # Mirroring, usually between green and red channels
    do_mirroring: bool = False

    # Rigid alignment (slices to each other)
    do_rigid_alignment: bool = False

    # Datatypes and scaling
    initial_dtype: str = 'uint16'  # Filtering etc. will act on this
    final_dtype: str = 'uint8'
    alpha: float = 0.15

    # Load results of a separate preprocessing step, if available
    to_save_warp_matrices: bool = True
    to_use_previous_warp_matrices: bool = False
    path_to_previous_warp_matrices: str = None  # This file may not exist, and will be written the first time

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, 'r') as f:
            cfg = yaml.safe_load(f)
        return PreprocessingSettings(**cfg)

    def write_to_yaml(self, fname):
        edit_config(fname, dataclasses.asdict(self))



def perform_preprocessing(dat_raw: typing.Union[np.ndarray, zarr.Array],
                          preprocessing_settings: PreprocessingSettings) -> np.ndarray:
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings
    if s is None:
        return dat_raw

    if s.do_filtering:
        dat_raw = filter_stack(dat_raw, s.filter_opt)

    if s.do_mirroring:
        dat_raw = np.flip(dat_raw, axis=2)

    if s.do_rigid_alignment:
        if not s.to_use_previous_warp_matrices:
            dat_raw, warp_matrices_dict = align_stack(dat_raw, s.to_save_warp_matrices)
            if s.to_save_warp_matrices:
                with open(s.path_to_previous_warp_matrices, 'wb') as f:
                    pickle.dump(warp_matrices_dict, f)
        else:
            with open(s.path_to_previous_warp_matrices, 'rb') as f:
                warp_matrices_dict = pickle.load(f)
            dat_raw = align_stack_using_previous_results(dat_raw, warp_matrices_dict)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        dat_raw = ndi.maximum_filter(dat_raw, size=(mini_max_size, 1, 1))

    dat_raw = (dat_raw*s.alpha).astype(s.final_dtype)

    return dat_raw


def preprocess_all_frames_using_config(DEBUG: bool, config: dict, verbose: int, vid_fname: str,
                                       preprocessing_settings: PreprocessingSettings = None,
                                       which_frames: list = None) -> Tuple[zarr.Array, dict]:
    """
    Preproceses all frames that will be analyzed as per config

    NOTE: expects 'preprocessing_config' and 'dataset_params' to be in config
    OR for the PreprocessingSettings object to be passed directly

    Loads but does not process frames before config['dataset_params']['start_volume']
        (to keep the indices the same as the original dataset)
    """
    if preprocessing_settings is None:
        p = PreprocessingSettings.load_from_yaml(config['preprocessing_config'])
    else:
        p = preprocessing_settings
        
    num_slices, num_total_frames, start_volume, sz, vid_opt = _preprocess_all_frames_unpack_config(config, verbose,
                                                                                                   vid_fname)
    return preprocess_all_frames(DEBUG, num_slices, num_total_frames, p, start_volume, sz, vid_fname, vid_opt,
                                 which_frames)


def preprocess_all_frames(DEBUG: bool, num_slices: int, num_total_frames: int, p: PreprocessingSettings,
                          start_volume: int, sz: Tuple, vid_fname: str, vid_opt: dict,
                          which_frames: list) -> Tuple[zarr.Array, dict]:
    if DEBUG:
        # Make a much shorter video
        if which_frames is not None:
            num_total_frames = which_frames[-1] + 1
        else:
            num_total_frames = 2
    chunk_sz = (1, num_slices,) + sz
    total_sz = (num_total_frames,) + chunk_sz[1:]

    preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype='uint16',
                                  synchronizer=zarr.ThreadSynchronizer())
    read_lock = threading.Lock()
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    with tifffile.TiffFile(vid_fname) as vid_stream:
        with tqdm(total=num_total_frames) as pbar:
            def parallel_func(i):
                preprocessed_dat[i, ...] = get_and_preprocess(i, num_slices, p, start_volume, vid_stream, read_lock)
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                # futures = executor.map(parallel_func, frame_list)
                # [f.result() for f in futures]
                futures = {executor.submit(parallel_func, i): i for i in frame_list}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)
        # for i in tqdm(frame_list):
        #     preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream)
    return preprocessed_dat, vid_opt


def _preprocess_all_frames_unpack_config(config, verbose, vid_fname):
    sz, vid_opt = _get_video_options(config, vid_fname)
    if verbose >= 1:
        print("Preprocessing data, this could take a while...")
    start_volume = config['dataset_params']['start_volume']
    num_total_frames = start_volume + config['dataset_params']['num_frames']
    num_slices = config['dataset_params']['num_slices']
    return num_slices, num_total_frames, start_volume, sz, vid_opt


def _get_video_options(config, vid_fname):
    with tifffile.TiffFile(vid_fname) as tif:
        sz = tif.pages[0].shape
    vid_opt = {'fps': config['dataset_params']['fps'],
               'frame_height': sz[0],
               'frame_width': sz[1],
               'is_color': False}
    if 'training_data_2d' in config:
        vid_opt['is_color'] = config['training_data_2d'].get('is_color', False)
    return sz, vid_opt


def get_and_preprocess(i, num_slices, p, start_volume, vid_fname, read_lock=None):
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