import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import zarr

from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


@dataclass
class finished_project_data:
    project_dir: str

    red_data: zarr.Array
    green_data: zarr.Array

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    final_tracks: pd.DataFrame

    raw_segmentation: zarr.Array
    segmentation: zarr.Array

    behavior_annotations: pd.DataFrame
    background_per_pixel: float

    verbose: int = 2

    @staticmethod
    def unpack_config_file(project_path):
        cfg = modular_project_config(project_path)
        project_dir = cfg.project_dir

        segment_cfg = cfg.get_segmentation_config()
        tracking_cfg = cfg.get_tracking_config()
        traces_cfg = cfg.get_traces_config()
        # # project_dir = Path(project_path).parent
        # cfg = load_config(project_path)
        # with safe_cd(project_dir):
        #     traces_cfg = load_config(cfg['subfolder_configs']['traces'])
        #     segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
        #     tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])

        return cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def _load_data_from_configs(cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir):
        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.config['traces']['red']
        green_traces_fname = traces_cfg.config['traces']['green']
        final_tracks_fname = tracking_cfg.config['final_3d_tracks']['df_fname']
        seg_fname_raw = segment_cfg.config['output']['masks']
        seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')

        fname = r"3-tracking\postprocessing\manual_behavior_annotation.xlsx"  # TODO: do not hardcode

        red_data = zarr.open(red_dat_fname)
        green_data = zarr.open(green_dat_fname)

        with safe_cd(project_dir):
            red_traces = pd.read_hdf(red_traces_fname)
            green_traces = pd.read_hdf(green_traces_fname)
            final_tracks = pd.read_hdf(final_tracks_fname)

            # Segmentation
            if '.zarr' in seg_fname_raw:
                raw_segmentation = zarr.open(seg_fname_raw, mode='r')
            else:
                raw_segmentation = None

            if os.path.exists(seg_fname):
                segmentation = zarr.open(seg_fname, mode='r')
            else:
                segmentation = None

            behavior_annotations = pd.read_excel(fname, sheet_name='behavior')['Annotation']

        # TODO: do not hardcode
        background_per_pixel = 14

        start = cfg.config['dataset_params']['start_volume']
        end = start + cfg.config['dataset_params']['num_frames']
        x = list(range(start, end))

        # Return a full object
        obj = finished_project_data(
            project_dir,
            red_data,
            green_data,
            red_traces,
            green_traces,
            final_tracks,
            raw_segmentation,
            segmentation,
            behavior_annotations,
            background_per_pixel
        )

        return obj

    @staticmethod
    def load_final_project_data_from_config(project_path):
        if isinstance(project_path, (str, os.PathLike)):
            args = finished_project_data.unpack_config_file(project_path)
            return finished_project_data._load_data_from_configs(*args)
        elif isinstance(project_path, finished_project_data):
            return project_path
        else:
            raise TypeError("Must path pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str, calculation_mode: str, neuron_name: str):
        assert (channel_mode in ['green', 'red', 'ratio']), f"Unknown channel mode {channel_mode}"

        if self.verbose >= 1:
            print(f"Calculating {channel_mode} trace for {neuron_name} for {calculation_mode} mode")

        # Way to process a single dataframe
        if calculation_mode == 'integration':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['brightness']
                return y_raw - self.background_per_pixel * df_tmp[i]['volume']
        elif calculation_mode == 'max':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['all_values']
                f = lambda x: np.max(x, initial=np.nan)
                return y_raw.apply(f) - self.background_per_pixel
        elif calculation_mode == 'mean':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['brightness']
                return y_raw / df_tmp[i]['volume'] - self.background_per_pixel
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

    def shade_axis_using_behavior(self, ax):
        shade_using_behavior(self.behavior_annotations, ax)
