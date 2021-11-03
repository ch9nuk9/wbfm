import concurrent
import logging
import os
from dataclasses import dataclass
from typing import Tuple

import napari
import numpy as np
import pandas as pd
import zarr

from DLC_for_WBFM.utils.visualization.filtering_traces import remove_outliers_via_rolling_mean, filter_rolling_mean, \
    filter_linear_interpolation, trace_from_dataframe_factory
from segmentation.util.utils_metadata import DetectedNeurons
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, read_if_exists, pickle_load_binary, \
    SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_project import safe_cd


@dataclass
class ProjectData:
    project_dir: str
    project_config: ModularProjectConfig

    red_data: zarr.Array
    green_data: zarr.Array

    raw_segmentation: zarr.Array
    segmentation: zarr.Array
    segmentation_metadata: DetectedNeurons

    df_training_tracklets: pd.DataFrame
    reindexed_masks_training: zarr.Array
    reindexed_metadata_training: DetectedNeurons

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    final_tracks: pd.DataFrame

    behavior_annotations: pd.DataFrame
    background_per_pixel: float
    likelihood_thresh: float

    verbose: int = 2

    _raw_frames: dict = None
    _raw_matches: dict = None
    _raw_clust: pd.DataFrame = None
    _df_all_tracklets: pd.DataFrame = None
    _df_fdnc_tracks: pd.DataFrame = None

    # Can be quite large, so don't read by default
    @property
    def raw_frames(self):
        if self._raw_frames is None:
            logging.info("First time loading the raw frames, may take a while...")
            train_cfg = self.project_config.get_training_config()
            fname = os.path.join('raw', 'frame_dat.pickle')
            fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            frames = pickle_load_binary(fname)
            self._raw_frames = frames
        return self._raw_frames

    @property
    def raw_matches(self):
        if self._raw_matches is None:
            logging.info("First time loading the raw matches, may take a while...")
            train_cfg = self.project_config.get_training_config()
            fname = os.path.join('raw', 'match_dat.pickle')
            fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            matches = pickle_load_binary(fname)
            self._raw_matches = matches
        return self._raw_matches

    @property
    def raw_clust(self):
        if self._raw_clust is None:
            logging.info("First time loading the raw cluster dataframe, may take a while...")
            train_cfg = self.project_config.get_training_config()
            fname = os.path.join('raw', 'clust_df_dat.pickle')
            fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            clust = pickle_load_binary(fname)
            self._raw_clust = clust
        return self._raw_clust

    @property
    def df_all_tracklets(self):
        if self._df_all_tracklets is None:
            train_cfg = self.project_config.get_training_config()
            fname = train_cfg.resolve_relative_path_from_config('df_3d_tracklets')
            self._df_all_tracklets = read_if_exists(fname)
        return self._df_all_tracklets

    @property
    def df_fdnc_tracks(self):
        if self._df_fdnc_tracks is None:
            train_cfg = self.project_config.get_tracking_config()
            fname = os.path.join('postprocessing', 'leifer_tracks.h5')
            fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            self._df_fdnc_tracks = read_if_exists(fname)
        return self._df_fdnc_tracks

    @property
    def num_frames(self):
        return self.project_config.config['dataset_params']['num_frames']

    @property
    def which_training_frames(self):
        # TODO: change this to just load on init?
        train_cfg = self.project_config.get_training_config()
        return train_cfg.config['training_data_3d']['which_frames']

    @staticmethod
    def unpack_config_file(project_path):
        cfg = ModularProjectConfig(project_path)
        project_dir = cfg.project_dir

        segment_cfg = cfg.get_segmentation_config()
        train_cfg = cfg.get_training_config()
        tracking_cfg = cfg.get_tracking_config()
        traces_cfg = cfg.get_traces_config()

        return cfg, segment_cfg, train_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def _load_data_from_configs(cfg: ModularProjectConfig,
                                segment_cfg: SubfolderConfigFile,
                                train_cfg: SubfolderConfigFile,
                                tracking_cfg: SubfolderConfigFile,
                                traces_cfg: SubfolderConfigFile,
                                project_dir):
        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['red'])
        green_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['green'])

        df_training_tracklets_fname = train_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
        reindexed_masks_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_masks')

        final_tracks_fname = tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
        seg_fname_raw = segment_cfg.resolve_relative_path_from_config('output_masks')
        seg_fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')

        # Metadata uses class from segmentation package
        seg_metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        seg_metadata = DetectedNeurons(seg_metadata_fname)
        reindexed_metadata_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_metadata')
        reindexed_metadata_training = DetectedNeurons(reindexed_metadata_training_fname)

        # TODO: do not hardcode
        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"
        behavior_fname = cfg.resolve_relative_path(behavior_fname)

        zarr_reader = lambda fname: zarr.open(fname, mode='r')
        excel_reader = lambda fname: pd.read_excel(fname, sheet_name='behavior')['Annotation']

        # Note: when running on the cluster the raw data isn't (for now) accessible
        # , reader=zarr_reader)

        with safe_cd(cfg.project_dir):

            logging.info("Starting threads to read data...")
            with concurrent.futures.ThreadPoolExecutor() as ex:
                red_data = ex.submit(read_if_exists, red_dat_fname, zarr_reader).result()
                green_data = ex.submit(read_if_exists, green_dat_fname, zarr_reader).result()
                red_traces = ex.submit(read_if_exists, red_traces_fname).result()
                green_traces = ex.submit(read_if_exists, green_traces_fname).result()
                df_training_tracklets = ex.submit(read_if_exists, df_training_tracklets_fname).result()
                reindexed_masks_training = ex.submit(read_if_exists, reindexed_masks_training_fname, zarr_reader).result()
                # reindexed_metadata_training = ex.submit(read_if_exists,
                #                                         reindexed_metadata_training_fname, pickle_load_binary).result()
                final_tracks = ex.submit(read_if_exists, final_tracks_fname).result()
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader).result()
                # seg_metadata: dict = ex.submit(pickle_load_binary, seg_metadata_fname).result()
                behavior_annotations = ex.submit(read_if_exists, behavior_fname, excel_reader).result()

            if red_traces is not None:
                red_traces.replace(0, np.nan, inplace=True)
                green_traces.replace(0, np.nan, inplace=True)
            logging.info("Read all data")

        background_per_pixel = traces_cfg.config['visualization']['background_per_pixel']
        likelihood_thresh = traces_cfg.config['visualization']['likelihood_thresh']

        start = cfg.config['dataset_params']['start_volume']
        end = start + cfg.config['dataset_params']['num_frames']
        x = list(range(start, end))

        # Return a full object
        obj = ProjectData(
            cfg.project_dir,
            cfg,
            red_data,
            green_data,
            raw_segmentation,
            segmentation,
            seg_metadata,
            df_training_tracklets,
            reindexed_masks_training,
            reindexed_metadata_training,
            red_traces,
            green_traces,
            final_tracks,
            behavior_annotations,
            background_per_pixel,
            likelihood_thresh
        )
        print(obj)

        return obj

    @staticmethod
    def load_final_project_data_from_config(project_path):
        if isinstance(project_path, (str, os.PathLike)):
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args)
        elif isinstance(project_path, ModularProjectConfig):
            project_path = project_path.self_path
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args)
        elif isinstance(project_path, ProjectData):
            return project_path
        else:
            raise TypeError("Must pass pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str, calculation_mode: str, neuron_name: str,
                         remove_outliers: bool = False,
                         filter_mode: str = 'no_filtering',
                         min_confidence: float = None):
        assert (channel_mode in ['green', 'red', 'ratio']), f"Unknown channel mode {channel_mode}"

        if self.verbose >= 3:
            print(f"Calculating {channel_mode} trace for {neuron_name} for {calculation_mode} mode")

        calc_single_trace = trace_from_dataframe_factory(calculation_mode, self.background_per_pixel)

        # How to combine channels, or which channel to choose
        if channel_mode in ['red', 'green']:
            if channel_mode == 'red':
                df = self.red_traces
            else:
                df = self.green_traces

            def calc_y(i):
                return calc_single_trace(i, df)
        else:
            df_red = self.red_traces
            df_green = self.green_traces

            def calc_y(i):
                return calc_single_trace(i, df_green) / calc_single_trace(i, df_red)

        y = calc_y(neuron_name)

        # Then remove outliers and / or filter
        if min_confidence is not None:
            low_confidence = self.final_tracks[neuron_name]['likelihood'] < min_confidence
            nan_confidence = np.isnan(self.final_tracks[neuron_name]['likelihood'])
            outliers_from_tracking = np.logical_or(low_confidence, nan_confidence)
            y[outliers_from_tracking] = np.nan

        # TODO: allow parameter selection
        if remove_outliers:
            y = remove_outliers_via_rolling_mean(y, window=9)

        # TODO: set up enum
        if filter_mode == "rolling_mean":
            y = filter_rolling_mean(y, window=5)
        elif filter_mode == "linear_interpolation":
            y = filter_linear_interpolation(y, window=15)
        elif filter_mode == "no_filtering":
            pass
        else:
            logging.warning(f"Unrecognized filter mode: {filter_mode}")

        return y


def napari_of_training_data(cfg: ModularProjectConfig) -> Tuple[napari.Viewer, np.ndarray, np.ndarray]:

    project_data = ProjectData.load_final_project_data_from_config(cfg)
    training_cfg = cfg.get_training_config()

    z_dat = project_data.red_data
    raw_seg = project_data.raw_segmentation
    z_seg = project_data.reindexed_masks_training

    # Training data doesn't usually start at i=0, so align
    num_frames = training_cfg.config['training_data_3d']['num_training_frames']
    i_seg_start = training_cfg.config['training_data_3d']['which_frames'][0]
    i_seg_end = i_seg_start + num_frames
    z_dat = z_dat[i_seg_start:i_seg_end, ...]
    raw_seg = raw_seg[i_seg_start:i_seg_end, ...]

    logging.info(f"Size of reindexed_masks: {z_dat.shape}")

    viewer = napari.view_labels(z_seg, ndisplay=3)
    viewer.add_labels(raw_seg, visible=False)
    viewer.add_image(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg
