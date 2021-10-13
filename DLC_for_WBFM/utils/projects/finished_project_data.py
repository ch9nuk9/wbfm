import concurrent
import logging
import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, read_if_exists, pickle_load_binary
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


@dataclass
class finished_project_data:
    project_dir: str
    project_config: modular_project_config

    red_data: zarr.Array
    green_data: zarr.Array

    raw_segmentation: zarr.Array
    segmentation: zarr.Array
    segmentation_metadata: dict

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    final_tracks: pd.DataFrame

    behavior_annotations: pd.DataFrame
    background_per_pixel: float
    likelihood_thresh: float

    verbose: int = 2

    @property
    def raw_frames(self):
        fname = Path(self.project_dir).joinpath('2-training_data').joinpath('raw').joinpath('frame_dat.pickle')
        with open(fname, 'rb') as f:
            frames = pickle.load(f)
        return frames

    @property
    def raw_matches(self):
        fname = Path(self.project_dir).joinpath('2-training_data').joinpath('raw').joinpath('match_dat.pickle')
        with open(fname, 'rb') as f:
            matches = pickle.load(f)
        return matches

    @property
    def num_frames(self):
        return self.project_config.config['dataset_params']['num_frames']

    @staticmethod
    def unpack_config_file(project_path):
        cfg = modular_project_config(project_path)
        project_dir = cfg.project_dir

        segment_cfg = cfg.get_segmentation_config()
        tracking_cfg = cfg.get_tracking_config()
        traces_cfg = cfg.get_traces_config()

        return cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def _load_data_from_configs(cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir):
        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.config['traces']['red']
        green_traces_fname = traces_cfg.config['traces']['green']
        final_tracks_fname = tracking_cfg.config['final_3d_tracks_df']
        seg_fname_raw = segment_cfg.config['output_masks']
        seg_metadata_fname = segment_cfg.config['output_metadata']
        seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')

        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"  # TODO: do not hardcode

        red_data = zarr.open(red_dat_fname)
        green_data = zarr.open(green_dat_fname)

        with safe_cd(project_dir):
            zarr_reader = lambda fname: zarr.open(fname, mode='r')
            excel_reader = lambda fname: pd.read_excel(fname, sheet_name='behavior')['Annotation']

            logging.info("Starting threads to read data...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f = executor.submit(read_if_exists, red_traces_fname)
                red_traces = f.result()
                f = executor.submit(read_if_exists, green_traces_fname)
                green_traces = f.result()
                f = executor.submit(read_if_exists, final_tracks_fname)
                final_tracks = f.result()
                f = executor.submit(read_if_exists, seg_fname_raw, zarr_reader)
                raw_segmentation = f.result()
                f = executor.submit(read_if_exists, seg_fname, zarr_reader)
                segmentation = f.result()
                f = executor.submit(pickle_load_binary, seg_metadata_fname)
                seg_metadata: dict = f.result()
                f = executor.submit(read_if_exists, behavior_fname, excel_reader)
                behavior_annotations = f.result()

            logging.info("Read all data")

        background_per_pixel = traces_cfg.config['visualization']['background_per_pixel']
        likelihood_thresh = traces_cfg.config['visualization']['likelihood_thresh']

        start = cfg.config['dataset_params']['start_volume']
        end = start + cfg.config['dataset_params']['num_frames']
        x = list(range(start, end))

        # Return a full object
        obj = finished_project_data(
            project_dir,
            cfg,
            red_data,
            green_data,
            raw_segmentation,
            segmentation,
            seg_metadata,
            red_traces,
            green_traces,
            final_tracks,
            behavior_annotations,
            background_per_pixel,
            likelihood_thresh
        )

        return obj

    @staticmethod
    def load_final_project_data_from_config(project_path):
        if isinstance(project_path, (str, os.PathLike)):
            args = finished_project_data.unpack_config_file(project_path)
            return finished_project_data._load_data_from_configs(*args)
        elif isinstance(project_path, modular_project_config):
            project_path = project_path.project_path
            args = finished_project_data.unpack_config_file(project_path)
            return finished_project_data._load_data_from_configs(*args)
        elif isinstance(project_path, finished_project_data):
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
        vol0_zxy = self.segmentation_metadata[i_frame]['centroids'].to_numpy()
        return np.array([np.array(m) for m in vol0_zxy])

    def __repr__(self):
        return f"=======================================\n\
Project data:\n\
project_dir: {self.project_dir} \n\
=======================================\n\
Found data:\n\
red_data: {self.red_data is not None}\n\
green_data: {self.green_data is not None}\n\
raw_segmentation: {self.raw_segmentation is not None}\n\
segmentation: {self.segmentation is not None}\n\
red_traces: {self.red_traces is not None}\n\
green_traces: {self.green_traces is not None}\n\
final_tracks: {self.final_tracks is not None}\n\
behavior_annotations: {self.behavior_annotations is not None}\n"
