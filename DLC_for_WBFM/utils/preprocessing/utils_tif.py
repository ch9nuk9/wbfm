import concurrent
import dataclasses
import pickle
import threading
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple
import numpy as np
import scipy.ndimage as ndi
# import tifffile
from ruamel.yaml import YAML
import zarr
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.feature_detection.utils_rigid_alignment import align_stack, filter_stack, \
    align_stack_using_previous_results
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

    # Plane removal, especially flyback
    raw_number_of_planes: int = None
    starting_plane: int = None

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

    # Temporary variable to store warp matrices across time
    all_warp_matrices: dict = field(default_factory=dict)

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, 'r') as f:
            cfg = YAML().load(f)
        return PreprocessingSettings(**cfg)

    def write_to_yaml(self, fname):
        edit_config(fname, dataclasses.asdict(self))

    def save_all_warp_matrices(self):
        with open(self.path_to_previous_warp_matrices, 'wb') as f:
            pickle.dump(self.all_warp_matrices, f)

    def load_all_warp_matrices(self):
        """
        Loads warp matrices from disk

        Note that if the same preprocessing_settings object is used to process multiple files, then this is not needed
        """
        with open(self.path_to_previous_warp_matrices, 'rb') as f:
            self.all_warp_matrices = pickle.load(f)


def perform_preprocessing(single_volume_raw: np.ndarray,
                          preprocessing_settings: PreprocessingSettings,
                          which_frame: int = None) -> np.ndarray:
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings
    if s is None:
        return single_volume_raw

    if s.starting_plane is not None:
        single_volume_raw = single_volume_raw[s.starting_plane:, ...]

    if s.do_filtering:
        single_volume_raw = filter_stack(single_volume_raw, s.filter_opt)

    if s.do_mirroring:
        single_volume_raw = np.flip(single_volume_raw, axis=-1)

    if s.do_rigid_alignment:
        if not s.to_use_previous_warp_matrices:
            single_volume_raw, warp_matrices_dict = align_stack(single_volume_raw)
            if s.to_save_warp_matrices:
                s.all_warp_matrices[which_frame] = warp_matrices_dict
        else:
            assert len(s.all_warp_matrices) > 0
            warp_matrices_dict = s.all_warp_matrices[which_frame]
            single_volume_raw = align_stack_using_previous_results(single_volume_raw, warp_matrices_dict)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        single_volume_raw = ndi.maximum_filter(single_volume_raw, size=(mini_max_size, 1, 1))

    single_volume_raw = (single_volume_raw * s.alpha).astype(s.final_dtype)

    return single_volume_raw


def preprocess_all_frames_using_config(DEBUG: bool, config: dict, verbose: int, video_fname: str,
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
                                                                                                   video_fname)
    return preprocess_all_frames(DEBUG, num_slices, num_total_frames, p, start_volume, sz, video_fname, vid_opt,
                                 which_frames)


def preprocess_all_frames(DEBUG: bool, num_slices: int, num_total_frames: int, p: PreprocessingSettings,
                          start_volume: int, sz: Tuple, video_fname: str, vid_opt: dict,
                          which_frames: list) -> Tuple[zarr.Array, dict]:
    import tifffile

    if DEBUG:
        # Make a much shorter video
        if which_frames is not None:
            num_total_frames = which_frames[-1] + 1
        else:
            num_total_frames = 2
        print("DEBUG MODE: Applying preprocessing:")
        print(p)
    chunk_sz = (1, num_slices,) + sz
    total_sz = (num_total_frames,) + chunk_sz[1:]

    preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype=p.final_dtype,
                                  synchronizer=zarr.ThreadSynchronizer())
    read_lock = threading.Lock()
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    with tifffile.TiffFile(video_fname) as vid_stream:
        with tqdm(total=num_total_frames) as pbar:
            def parallel_func(i):
                preprocessed_dat[i, ...] = get_and_preprocess(i, num_slices, p, start_volume, vid_stream,
                                                              read_lock)

            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                futures = {executor.submit(parallel_func, i): i for i in frame_list}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)
        # for i in tqdm(frame_list):
        #     preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream)
    return preprocessed_dat, vid_opt


def _preprocess_all_frames_unpack_config(config, verbose, video_fname):
    sz, vid_opt = _get_video_options(config, video_fname)
    if verbose >= 1:
        print("Preprocessing data, this could take a while...")
    start_volume = config['dataset_params']['start_volume']
    num_total_frames = start_volume + config['dataset_params']['num_frames']
    num_slices = config['dataset_params']['num_slices']
    return num_slices, num_total_frames, start_volume, sz, vid_opt


def _get_video_options(config, video_fname):
    import tifffile
    if video_fname.endswith('.tif') or video_fname.endswith('.btf'):
        with tifffile.TiffFile(video_fname) as tif:
            sz = tif.pages[0].shape
    elif video_fname.endswith('.zarr'):
        sz = zarr.open(video_fname).shape[2:]
    else:
        raise FileNotFoundError("Must pass .zarr or .tif or .btf file")
    vid_opt = {'fps': config['dataset_params']['fps'],
               'frame_height': sz[0],
               'frame_width': sz[1],
               'is_color': False}
    if 'training_data_2d' in config:
        vid_opt['is_color'] = config['training_data_2d'].get('is_color', False)
    return sz, vid_opt


def get_and_preprocess(i, num_slices, p, start_volume, video_fname, read_lock=None):
    if p.raw_number_of_planes is not None:
        num_slices = p.raw_number_of_planes
    if read_lock is None:
        single_volume_raw = get_single_volume(video_fname, i, num_slices, dtype='uint16')
    else:
        with read_lock:
            single_volume_raw = get_single_volume(video_fname, i, num_slices, dtype='uint16')
    # Don't preprocess data that we didn't even segment!
    if i >= start_volume:
        return perform_preprocessing(single_volume_raw, p, i)
    else:
        return single_volume_raw
