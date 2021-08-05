import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import zarr

from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd


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

    @staticmethod
    def unpack_config_file(project_path):
        project_dir = Path(project_path).parent
        cfg = load_config(project_path)
        with safe_cd(project_dir):
            traces_cfg = load_config(cfg['subfolder_configs']['traces'])
            segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
            tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])

        return cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def load_data_from_configs(cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir):
        red_dat_fname = cfg['preprocessed_red']
        green_dat_fname = cfg['preprocessed_green']
        red_traces_fname = traces_cfg['traces']['red']
        green_traces_fname = traces_cfg['traces']['green']
        final_tracks_fname = tracking_cfg['final_3d_tracks']['df_fname']
        seg_fname_raw = segment_cfg['output']['masks']
        seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')

        fname = r"3-tracking\postprocessing\manual_behavior_annotation.xlsx"  # TODO

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

        start = cfg['dataset_params']['start_volume']
        end = start + cfg['dataset_params']['num_frames']
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
    def load_all_project_data_from_config(project_path):
        if isinstance(project_path, (str, os.PathLike)):
            args = finished_project_data.unpack_config_file(project_path)
            return finished_project_data.load_data_from_configs(*args)
        elif isinstance(project_path, finished_project_data):
            return project_path
        else:
            raise TypeError("Must path pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str, calculation_mode: str, neuron_name: str):
        assert (channel_mode in ['green', 'red', 'ratio']), f"Unknown channel mode {channel_mode}"

        # Way to process a single dataframe
        if calculation_mode is 'integration':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['brightness']
                return y_raw - self.background_per_pixel * df_tmp[i]['volume']
        elif calculation_mode is 'max':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['all_values']
                return np.max(y_raw) - self.background_per_pixel
        elif calculation_mode is 'mean':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['brightness']
                return y_raw / df_tmp[i]['volume'] - self.background_per_pixel
        elif calculation_mode is 'quantile90':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['all_values']
                return np.quantile(y_raw, 0.9) - self.background_per_pixel
        elif calculation_mode is 'quantile50':
            def calc_single_trace(i, df_tmp):
                y_raw = df_tmp[i]['all_values']
                return np.quantile(y_raw, 0.5) - self.background_per_pixel
        else:
            raise ValueError(f"Unknown calculation mode {calculation_mode}")

        # How to combine channels, or which channel to choose
        if channel_mode in ['red', 'green']:
            if channel_mode is 'red':
                df = self.red_traces
            else:
                df = self.green_traces
            def calc_y(i):
                calc_single_trace(i, df)
        else:
            df_red = self.red_traces
            df_green = self.green_traces
            def calc_y(i):
                return calc_single_trace(i, df_green) / calc_single_trace(i, df_red)

        return calc_y(neuron_name)
