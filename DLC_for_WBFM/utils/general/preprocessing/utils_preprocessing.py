import concurrent
import concurrent.futures
import dataclasses
import logging
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import dask.array
import numpy as np
import zarr
from backports.cached_property import cached_property
from ruamel.yaml import YAML
from scipy import ndimage as ndi
from tifffile import tifffile
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from DLC_for_WBFM.utils.neuron_matching.utils_rigid_alignment import filter_stack, align_stack_to_middle_slice, \
    align_stack_using_previous_results, apply_alignment_matrix_to_stack
from DLC_for_WBFM.utils.projects.paths_to_external_resources import get_camera_alignment_matrix
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, ConfigFileWithProjectContext
from DLC_for_WBFM.utils.projects.utils_filenames import add_name_suffix
from DLC_for_WBFM.utils.projects.utils_project import edit_config
from DLC_for_WBFM.utils.general.video_and_data_conversion.import_video_as_array import get_single_volume


def background_subtract_single_channel(raw_fname, background_fname, num_frames, num_slices, preprocessing_settings,
                                       DEBUG=False):

    raw_data = zarr_reader_folder_or_zipstore(raw_fname)
    background_video_list = read_background(background_fname, num_frames, num_slices,
                                            preprocessing_settings)
    # Add a new truly constant background value, to keep anything from going negative
    new_background = preprocessing_settings.background_per_pixel
    # Get a single image, because that's the physical camera
    background_video_mean = np.mean(np.mean(background_video_list, axis=0), axis=0) - new_background
    # Don't try to modify the data as read; it is read-only
    fname_subtracted = add_name_suffix(raw_fname, '_background_subtracted')
    logging.info(f"Creating data copy at {fname_subtracted}")
    store = zarr.DirectoryStore(fname_subtracted)
    background_subtracted = zarr.zeros_like(raw_data, store=store)
    # Loop so that not all is loaded in memory... should I use dask?
    for i, volume in enumerate(tqdm(raw_data)):
        background_subtracted[i, ...] = np.array(np.maximum(volume - background_video_mean, 0), dtype=volume.dtype)
        if DEBUG and i > 5:
            break
    # zarr.save_array(background_subtracted, fname_subtracted)

    return fname_subtracted


def read_background(background_fname, num_frames, num_slices, preprocessing_settings=None):
    background_video_list = []
    with tifffile.TiffFile(background_fname) as background_tiff:
        for i in tqdm(range(num_frames)):
            try:
                background_volume = get_single_volume(background_tiff, i, num_slices, dtype='uint16')
            except IndexError:
                break
            # Note: this will do rigid rotation
            background_volume = perform_preprocessing(background_volume, preprocessing_settings, i)
            background_video_list.append(background_volume)
    logging.info(f"Read background tiff file of shape: {background_video_list[0].shape}")
    return background_video_list


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

    do_background_subtraction: bool = True
    background_fname_red: str = None
    background_fname_green: str = None

    background_red: np.ndarray = None
    background_green: np.ndarray = None
    background_per_pixel: int = 100

    # Mini max
    do_mini_max_projection: bool = False
    mini_max_size: int = 3

    # Mirroring, usually between green and red channels
    do_mirroring: bool = False

    # Rigid alignment (slices to each other)
    do_rigid_alignment: bool = False

    # Rigid alignment (green to red channel)
    align_green_red_cameras: bool = False

    # Datatypes and scaling
    initial_dtype: str = 'uint16'  # Filtering etc. will act on this
    final_dtype: str = 'uint16'
    uint8_only_for_opencv: bool = True
    alpha: float = 0.15  # Deprecated

    alpha_red: float = None
    alpha_green: float = None

    # For updating self on disk
    cfg_preprocessing: ConfigFileWithProjectContext = None

    # Load results of a separate preprocessing step, if available
    to_save_warp_matrices: bool = True
    to_use_previous_warp_matrices: bool = False
    path_to_previous_warp_matrices: str = None  # This file may not exist, and will be written the first time

    # Temporary variable to store warp matrices across time
    all_warp_matrices: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.do_background_subtraction:
            self.initialize_background()
        if not self.alpha_is_ready:
            logging.warning("Alpha is not set in the yaml; will have to be calculated from data")

    @property
    def background_is_ready(self):
        if not self.do_background_subtraction:
            return True
        else:
            return self.background_red is not None

    def initialize_background(self):
        logging.info("Loading background videos, may take a minute")
        if self.background_fname_red is None:
            logging.debug("Didn't find background filename; this will have to be rerun")
            return

        self.background_red = self.load_background(self.background_fname_red)
        self.background_green = self.load_background(self.background_fname_green)

    def find_background_files_from_raw_data_path(self, cfg: ModularProjectConfig, force_search=False):
        if self.background_fname_red is not None:
            logging.info(f"Already has background at {self.background_fname_red}")
            if not force_search:
                return
        logging.info("Attempting to find new background files")

        folder_for_background = cfg.get_folder_with_background()

        for subfolder in folder_for_background.iterdir():

            subfolder_is_ch0 = ('background_Ch0' in subfolder.name) or ('background-channel-0' in subfolder.name)
            subfolder_is_ch1 = ('background_Ch1' in subfolder.name) or ('background-channel-1' in subfolder.name)

            if subfolder.is_dir() and subfolder_is_ch0:
                # Red channel
                for file in subfolder.iterdir():
                    if file.is_file() and 'background' in file.name and file.name.endswith('.btf'):
                        self.background_fname_red = str(file)
                        logging.info(f"Found red channel background at: {file}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find red background file within {folder_for_background}")
            elif subfolder.is_dir() and subfolder_is_ch1:
                # Green channel
                for file in subfolder.iterdir():
                    if file.is_file() and 'background' in file.name and file.name.endswith('.btf'):
                        self.background_fname_green = str(file)
                        logging.info(f"Found green channel background at: {file}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find green background file within {folder_for_background}")

        # Also update the preprocessing file on disk
        self.cfg_preprocessing.config['background_fname_red'] = self.background_fname_red
        self.cfg_preprocessing.config['background_fname_green'] = self.background_fname_green
        self.cfg_preprocessing.update_self_on_disk()

        # Actually load data
        self.initialize_background()

    def load_background(self, background_fname):
        num_frames = 40  # TODO
        background_video_list = read_background(background_fname, num_frames, self.raw_number_of_planes,
                                                preprocessing_settings=None)
        # Add a new truly constant background value, to keep anything from going negative
        new_background = self.background_per_pixel
        # Get a single image, because that's the physical camera
        background_video_mean = np.mean(np.mean(background_video_list, axis=0), axis=0)
        min_background_val = np.min(background_video_mean)
        if new_background > min_background_val:
            logging.warning(f"Can't set new background value to requested {new_background}, "
                            f"because some values would be negative. Using {min_background_val} instead")
            new_background = min_background_val
        background_video_mean -= new_background
        background_video_mean = background_video_mean.astype(self.initial_dtype)
        logging.info(f"Loaded background with mean: {np.mean(background_video_mean)}")

        return background_video_mean

    @property
    def alpha_is_ready(self):
        if self.alpha_red is not None and self.alpha_green is not None:
            return True
        else:
            return False

    def calculate_alpha_from_data(self, video, which_channel='red', num_volumes_to_load=None):
        # Note that this doesn't take into account background subtraction
        # Note: dask isn't faster than just numpy, but manages memory much better
        logging.info(f"Calculating alpha from data for channel {which_channel}; may take ~2 minutes per video")
        if num_volumes_to_load is None:
            # Load the entire video; only really works with zarr
            current_max_value = dask.array.from_zarr(video).max().compute()
            targeted_max_value = 254.0
        else:
            # Assumed to be tiff files; note that the stack axis doesn't matter
            # Requires setting an estimated maximum, to give clipping room
            this_dat = np.vstack([self.get_single_volume(video, i) for i in range(num_volumes_to_load)])
            current_max_value = np.max(this_dat)
            targeted_max_value = 200.0

        alpha = float(targeted_max_value / current_max_value)
        logging.info(f"Calculated alpha={alpha} for channel {which_channel}")

        if which_channel == 'red':
            self.alpha_red = alpha
            self.cfg_preprocessing.config['alpha_red'] = alpha
        elif which_channel == 'green':
            self.alpha_green = alpha
            self.cfg_preprocessing.config['alpha_green'] = alpha

        self.cfg_preprocessing.update_self_on_disk()

    @staticmethod
    def _load_from_yaml(fname, do_background_subtraction=None):
        with open(fname, 'r') as f:
            preprocessing_dict = YAML().load(f)
        if do_background_subtraction is not None:
            preprocessing_dict['do_background_subtraction'] = do_background_subtraction
        return PreprocessingSettings(**preprocessing_dict)

    @staticmethod
    def load_from_config(cfg: ModularProjectConfig, do_background_subtraction=None):
        fname = Path(cfg.project_dir).joinpath('preprocessing_config.yaml')
        preprocessing_settings = PreprocessingSettings._load_from_yaml(fname, do_background_subtraction)
        preprocessing_settings.cfg_preprocessing = cfg.get_preprocessing_config()
        if not preprocessing_settings.background_is_ready:
            try:
                preprocessing_settings.find_background_files_from_raw_data_path(cfg)
            except FileNotFoundError:
                logging.warning("Did not find background; turning off background subtraction")
                preprocessing_settings.do_background_subtraction = False
        return preprocessing_settings

    @cached_property
    def camera_alignment_matrix(self):
        warp_mat = get_camera_alignment_matrix()
        return warp_mat

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

    def get_single_volume(self, video_fname, i_time: int, num_slices=None):
        if num_slices is not None:
            raise NotImplementedError("Should set PreprocessingSettings.raw_number_of_planes")
        raw_volume = get_single_volume(video_fname, i_time, self.raw_number_of_planes, dtype=self.initial_dtype)
        return raw_volume


def perform_preprocessing(single_volume_raw: np.ndarray,
                          preprocessing_settings: PreprocessingSettings,
                          which_frame: int = None,
                          which_channel: str = 'red') -> np.ndarray:
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings
    if s is None:
        return single_volume_raw
    if s.final_dtype == 'uint8':
        logging.warning("This dtype should be uint16, not uint8. For now, it is ignored")

    if which_channel == 'red':
        alpha = s.alpha_red
        background = s.background_red
    elif which_channel == 'green':
        alpha = s.alpha_green
        background = s.background_green
    else:
        raise NotImplementedError(f"Unrecognized channel: {which_channel}")

    if s.starting_plane is not None:
        single_volume_raw = single_volume_raw[s.starting_plane:, ...]

    if s.do_background_subtraction:
        try:
            single_volume_raw = single_volume_raw - background
        except ValueError:
            logging.warning(f"The background {background.shape} was not the correct shape {single_volume_raw.shape}")
            logging.warning("Setting 'do_background_subtraction' to False")
            s.do_background_subtraction = False

    if s.do_filtering:
        single_volume_raw = filter_stack(single_volume_raw, s.filter_opt)

    if s.do_mirroring:
        single_volume_raw = np.flip(single_volume_raw, axis=-1)

    if s.do_rigid_alignment:
        if not s.to_use_previous_warp_matrices:
            try:
                single_volume_raw, warp_matrices_dict = align_stack_to_middle_slice(single_volume_raw)
            except cv2.error as e:
                logging.warning(f"When rigidly aligning in z, encountered opencv error {e}; leaving unaligned")
                warp_matrices_dict = {}
            if s.to_save_warp_matrices:
                s.all_warp_matrices[which_frame] = warp_matrices_dict
        else:
            assert len(s.all_warp_matrices) > 0
            warp_matrices_dict = s.all_warp_matrices[which_frame]
            if len(warp_matrices_dict) > 0:
                single_volume_raw = align_stack_using_previous_results(single_volume_raw, warp_matrices_dict)

    if s.align_green_red_cameras:
        alignment_mat = s.camera_alignment_matrix
        if alignment_mat is None:
            logging.warning("Requested red-green alignment, but no matrix was found")
        else:
            single_volume_raw = apply_alignment_matrix_to_stack(single_volume_raw, alignment_mat,
                                                                hide_progress=False)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        single_volume_raw = ndi.maximum_filter(single_volume_raw, size=(mini_max_size, 1, 1))

    # Do not actually change datatype
    if not s.uint8_only_for_opencv:
        raise DeprecationWarning("uint8 should not be saved directly, but converted on demand for opencv")

    return single_volume_raw


def preprocess_all_frames_using_config(DEBUG: bool, config: ModularProjectConfig, verbose: int, video_fname: str,
                                       preprocessing_settings: PreprocessingSettings = None,
                                       which_frames: list = None, which_channel: str = None,
                                       out_fname: str = None) -> Tuple[zarr.Array, dict]:
    """
    Preprocesses all frames that will be analyzed as per config

    NOTE: expects 'preprocessing_config' and 'dataset_params' to be in config
    OR for the PreprocessingSettings object to be passed directly

    Loads but does not process frames before config['dataset_params']['start_volume']
        (to keep the indices the same as the original dataset)
    """
    if preprocessing_settings is None:
        p = PreprocessingSettings.load_from_config(config)
    else:
        p = preprocessing_settings

    num_slices, num_total_frames, bigtiff_start_volume, sz, vid_opt = _preprocess_all_frames_unpack_config(config.config,
                                                                                                           verbose,
                                                                                                           video_fname)
    return preprocess_all_frames(DEBUG, num_slices, num_total_frames, p, bigtiff_start_volume, sz, video_fname, vid_opt,
                                 which_frames, which_channel, out_fname)


def preprocess_all_frames(DEBUG: bool, num_slices: int, num_total_frames: int, p: PreprocessingSettings,
                          start_volume: int, sz: Tuple, video_fname: str, vid_opt: dict,
                          which_frames: list, which_channel: str, out_fname: str) -> Tuple[zarr.Array, dict]:
    import tifffile
    if DEBUG:
        # Make a much shorter video
        if which_frames is not None:
            num_total_frames = which_frames[-1] + 1
        else:
            num_total_frames = 2
        print(p)

    chunk_sz = (1, num_slices,) + sz
    total_sz = (num_total_frames,) + chunk_sz[1:]
    store = zarr.DirectoryStore(path=out_fname)
    if p.final_dtype == np.uint8:
        raise DeprecationWarning("uint16 should be saved directly")
    preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype=p.initial_dtype,
                                  synchronizer=zarr.ThreadSynchronizer(),
                                  store=store)
    read_lock = threading.Lock()
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    with tifffile.TiffFile(video_fname) as vid_stream:
        # Note: this saves alpha to disk
        p.calculate_alpha_from_data(vid_stream, which_channel=which_channel, num_volumes_to_load=10)

        with tqdm(total=num_total_frames) as pbar:
            def parallel_func(i):
                preprocessed_dat[i, ...] = get_and_preprocess(i, p, start_volume, vid_stream,
                                                              which_channel, read_lock)

            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = {executor.submit(parallel_func, i): i for i in frame_list}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

    return preprocessed_dat, vid_opt


def _preprocess_all_frames_unpack_config(config: dict, verbose, video_fname):
    sz, vid_opt = _get_video_options(config, video_fname)
    if verbose >= 1:
        print("Preprocessing data, this could take a while...")
    start_volume = config['dataset_params'].get('bigtiff_start_volume', None)
    if start_volume is None:
        logging.warning("Did not find bigtiff_start_volume; is this an old style project?")
        logging.warning("Using start volume of 0. If this is fine, then no changes are needed")
        start_volume = 0
        config['dataset_params']['bigtiff_start_volume'] = 0  # Will be written to disk later
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


def get_and_preprocess(i, p, start_volume, video_fname, which_channel, read_lock=None):
    # Note: the preprocessing class must know the number of planes in a volume
    if read_lock is None:
        single_volume_raw = p.get_single_volume(video_fname, i)
    else:
        with read_lock:
            single_volume_raw = p.get_single_volume(video_fname, i)
    # Don't preprocess data that we didn't even segment!
    if i >= start_volume:
        return perform_preprocessing(single_volume_raw, p, i, which_channel=which_channel)
    else:
        return single_volume_raw
