import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from DLC_for_WBFM.utils.projects.utils_filepaths import lexigraphically_sort
from DLC_for_WBFM.utils.visualization.filtering_traces import trace_from_dataframe_factory, \
    remove_outliers_via_rolling_mean, filter_rolling_mean, filter_linear_interpolation


@dataclass
class TracePlotter:
    # Data
    red_traces: pd.DataFrame
    green_traces: pd.DataFrame
    final_tracks: pd.DataFrame

    # Settings
    channel_mode: str
    calculation_mode: str

    remove_outliers: bool = False
    filter_mode: str = 'no_filtering'
    min_confidence: float = None
    background_per_pixel: float = None

    verbose: int = 1

    def calculate_traces(self, neuron_name: str):
        assert (self.channel_mode in ['green', 'red', 'ratio']), f"Unknown channel mode {self.channel_mode}"

        if self.verbose >= 3:
            print(f"Calculating {self.channel_mode} trace for {neuron_name} for {self.calculation_mode} mode")

        calc_single_trace = trace_from_dataframe_factory(self.calculation_mode, self.background_per_pixel)

        # How to combine channels, or which channel to choose
        if self.channel_mode in ['red', 'green']:
            if self.channel_mode == 'red':
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
        if self.min_confidence is not None:
            low_confidence = self.final_tracks[neuron_name]['likelihood'] < self.min_confidence
            nan_confidence = np.isnan(self.final_tracks[neuron_name]['likelihood'])
            outliers_from_tracking = np.logical_or(low_confidence, nan_confidence)
            y[outliers_from_tracking] = np.nan

        # TODO: allow parameter selection
        if self.remove_outliers:
            y = remove_outliers_via_rolling_mean(y, window=9)

        # TODO: set up enum
        if self.filter_mode == "rolling_mean":
            y = filter_rolling_mean(y, window=5)
        elif self.filter_mode == "linear_interpolation":
            y = filter_linear_interpolation(y, window=15)
        elif self.filter_mode == "no_filtering":
            pass
        else:
            logging.warning(f"Unrecognized filter mode: {self.filter_mode}")

        return y


@dataclass
class TrackletPlotter:

    final_tracks: pd.DataFrame
    df_tracklets: pd.DataFrame
    global2tracklet: dict

    def calculate_tracklets_for_neuron(self, neuron_name) -> List[pd.DataFrame]:
        # Returns a list of pd.DataFrames with columns x, y, z, and likelihood, which can be plotted in a loop

        tracklet_ind = self.global2tracklet[neuron_name]
        all_tracklet_names = lexigraphically_sort(list(self.df_tracklets.columns.levels[0]))

        these_names = [all_tracklet_names[i] for i in tracklet_ind]
        these_tracklets = [self.df_tracklets[name] for name in these_names]

        return these_tracklets
