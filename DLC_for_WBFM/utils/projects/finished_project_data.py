import concurrent
import logging
import os
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
import zarr
from segmentation.util.utils_metadata import centroids_from_dict_of_dataframes
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, read_if_exists, pickle_load_binary, \
    ConfigFileWithProjectContext
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


@dataclass
class ProjectData:
    project_dir: str
    project_config: ModularProjectConfig

    red_data: zarr.Array
    green_data: zarr.Array

    raw_segmentation: zarr.Array
    segmentation: zarr.Array
    segmentation_metadata: dict

    df_training_tracklets: pd.DataFrame
    reindexed_masks_training: zarr.Array
    reindexed_metadata_training: dict

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    final_tracks: pd.DataFrame

    behavior_annotations: pd.DataFrame
    background_per_pixel: float
    likelihood_thresh: float

    verbose: int = 2

    # Can be quite large, so don't read by default
    @property
    def raw_frames(self):
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'frame_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        frames = pickle_load_binary(fname)
        return frames

    @property
    def raw_matches(self):
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'match_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        matches = pickle_load_binary(fname)
        return matches

    @property
    def df_all_tracklets(self):
        train_cfg = self.project_config.get_training_config()
        fname = train_cfg.resolve_relative_path_from_config('df_3d_tracklets')
        return read_if_exists(fname)

    @property
    def num_frames(self):
        return self.project_config.config['dataset_params']['num_frames']

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
                                segment_cfg: ConfigFileWithProjectContext,
                                train_cfg: ConfigFileWithProjectContext,
                                tracking_cfg: ConfigFileWithProjectContext,
                                traces_cfg: ConfigFileWithProjectContext,
                                project_dir):
        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.config['traces']['red']
        green_traces_fname = traces_cfg.config['traces']['green']

        df_training_tracklets_fname = train_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
        reindexed_masks_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_masks')
        reindexed_metadata_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_metadata')

        final_tracks_fname = tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
        seg_fname_raw = segment_cfg.resolve_relative_path_from_config('output_masks')
        seg_metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        seg_fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')

        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"  # TODO: do not hardcode

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
                reindexed_metadata_training = ex.submit(read_if_exists,
                                                        reindexed_metadata_training_fname, pickle_load_binary).result()
                final_tracks = ex.submit(read_if_exists, final_tracks_fname).result()
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader).result()
                seg_metadata: dict = ex.submit(pickle_load_binary, seg_metadata_fname).result()
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
            project_path = project_path.project_path
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args)
        elif isinstance(project_path, ProjectData):
            return project_path
        else:
            raise TypeError("Must pass pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str, calculation_mode: str, neuron_name: str):
        assert (channel_mode in ['green', 'red', 'ratio']), f"Unknown channel mode {channel_mode}"

        if self.verbose >= 1:
            print(f"Calculating {channel_mode} trace for {neuron_name} for {calculation_mode} mode")

        # Way to process a single dataframe
        if calculation_mode == 'integration':
            def calc_single_trace(i, df_tmp):
                try:
                    y_raw = df_tmp[i]['brightness']
                    vol = df_tmp[i]['volume']
                except KeyError:
                    y_raw = df_tmp[i]['intensity_image']
                    vol = df_tmp[i]['area']
                return y_raw - self.background_per_pixel * vol
        elif calculation_mode == 'max':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['all_values']
                f = lambda x: np.max(x, initial=np.nan)
                return y_raw.apply(f) - self.background_per_pixel
        elif calculation_mode == 'mean':
            def calc_single_trace(i, df_tmp):
                try:
                    y_raw = df_tmp[i]['brightness']
                    vol = df_tmp[i]['volume']
                except KeyError:
                    y_raw = df_tmp[i]['intensity_image']
                    vol = df_tmp[i]['area']
                return y_raw / vol - self.background_per_pixel
        # elif calculation_mode == 'quantile90':
        #     def calc_single_trace(i, df_tmp):
        #         y_raw = df_tmp[i]['all_values']
        #         return np.quantile(y_raw, 0.9) - self.background_per_pixel
        # elif calculation_mode == 'quantile50':
        #     def calc_single_trace(i, df_tmp):
        #         y_raw = df_tmp[i]['all_values']
        #         f = lambda x: np.quantile(x, initial=np.nan)
        #         return np.quantile(y_raw, 0.5) - self.background_per_pixel
        elif calculation_mode == 'volume':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['volume']
                return y_raw
        elif calculation_mode == 'z':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['z_dlc']
                return y_raw
        else:
            raise ValueError(f"Unknown calculation mode {calculation_mode}")

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

        return calc_y(neuron_name)

    def shade_axis_using_behavior(self, ax=None, behaviors_to_ignore='none'):
        if self.behavior_annotations is None:
            logging.warning("No behavior annotations present; skipping")
        else:
            shade_using_behavior(self.behavior_annotations, ax, behaviors_to_ignore)

    def get_centroids_as_numpy(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return centroids_from_dict_of_dataframes(self.segmentation_metadata, i_frame)

    def get_centroids_as_numpy_training(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return centroids_from_dict_of_dataframes(self.reindexed_metadata_training, i_frame)

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
=======================================\n\
Found the following data files:\n\
red_data: {self.red_data is not None}\n\
green_data: {self.green_data is not None}\n\
============Segmentation===============\n\
raw_segmentation: {self.raw_segmentation is not None}\n\
segmentation: {self.segmentation is not None}\n\
============Tracklets==================\n\
df_training_tracklets: {self.df_training_tracklets is not None}\n\
reindexed_masks_training: {self.reindexed_masks_training is not None}\n\
============Traces=====================\n\
red_traces: {self.red_traces is not None}\n\
green_traces: {self.green_traces is not None}\n\
final_tracks: {self.final_tracks is not None}\n\
behavior_annotations: {self.behavior_annotations is not None}\n"
