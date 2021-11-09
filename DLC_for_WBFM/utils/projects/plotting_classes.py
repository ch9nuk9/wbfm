import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
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
        if self.calculation_mode == 'likelihood':
            # Uses a different dataframe
            df = self.final_tracks

            def calc_y(i):
                return calc_single_trace(i, df)

        elif self.channel_mode in ['red', 'green']:
            if self.channel_mode == 'red':
                df = self.red_traces
            else:
                df = self.green_traces

            def calc_y(i):
                return calc_single_trace(i, df)
        elif self.channel_mode == 'ratio':
            df_red = self.red_traces
            df_green = self.green_traces

            def calc_y(i):
                return calc_single_trace(i, df_green) / calc_single_trace(i, df_red)
        else:
            raise ValueError("Unknown calculation or channel mode")

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
class TrackletAnnotator:

    df_tracklets: pd.DataFrame
    global2tracklet: dict

    # Annotation option
    manual_global2tracklet_names: dict = None
    current_neuron: str = None

    # Visualization options
    to_add_layer_to_viewer: bool = True
    verbose: int = 1

    def __post_init__(self):
        if self.manual_global2tracklet_names is None:
            self.manual_global2tracklet_names = defaultdict(list)

    def calculate_tracklets_for_neuron(self, neuron_name) -> List[pd.DataFrame]:
        # Returns a list of pd.DataFrames with columns x, y, z, and likelihood, which can be plotted in a loop

        tracklet_ind = self.global2tracklet[neuron_name]
        # all_tracklet_names = lexigraphically_sort(list(self.df_tracklets.columns.levels[0]))
        all_tracklet_names = list(self.df_tracklets.columns.levels[0])

        these_names = [all_tracklet_names[i] for i in tracklet_ind]
        if self.manual_global2tracklet_names is not None:
            these_names.extend(self.manual_global2tracklet_names[neuron_name])
        print(f"Found tracklets: {these_names}")
        these_tracklets = [self.df_tracklets[name] for name in these_names]

        return these_tracklets

    def connect_tracklet_clicking_callback(self, layer_to_add_callback, viewer,
                                           max_dist=10.0,
                                           refresh_callback=None):

        df_tracklets = self.df_tracklets

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):
            seg_index = layer.get_value(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True
            )

            if self.verbose >= 1:
                print(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                      f"and position {event.position[1:]}")

            dist, ind, tracklet_name = get_closest_tracklet_to_point(
                i_time=int(event.position[0]),
                target_pt=event.position[1:],
                df_tracklets=df_tracklets,
                verbose=2
            )

            if self.current_neuron is not None:
                self.manual_global2tracklet_names[self.current_neuron].append(tracklet_name)
                refresh_callback()

            dist = dist[0][0]
            if self.verbose >= 1:
                print(f"Neuron is part of tracklet {tracklet_name} with distance {dist}")

            if dist < max_dist:
                df_single_track = df_tracklets[tracklet_name]
                if self.verbose >= 1:
                    print(f"Adding tracklet of length {df_single_track['z'].count()}")
                if self.to_add_layer_to_viewer:
                    all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track)
                    viewer.add_tracks(track_of_point, name=tracklet_name)

                if self.verbose >= 2:
                    print(df_single_track.dropna(inplace=False))
            else:
                if self.verbose >= 1:
                    print(f"Tracklet too far away; not adding")


def get_closest_tracklet_to_point(i_time,
                                  target_pt,
                                  df_tracklets,
                                  nbr_obj: NearestNeighbors = None,
                                  nonnan_ind=None,
                                  verbose=0):
    # target_pt = df_tracks[which_neuron].iloc[i_time][:3]
    all_tracklet_names = lexigraphically_sort(list(df_tracklets.columns.levels[0]))

    if any(np.isnan(target_pt)):
        dist, ind_global_coords, tracklet_name = np.inf, None, None
    else:
        if nbr_obj is None:
            all_zxy = np.reshape(df_tracklets.iloc[i_time, :].to_numpy(), (-1, 4))
            nonnan_ind = ~np.isnan(all_zxy).any(axis=1)
            all_zxy = all_zxy[nonnan_ind][:, :3]
            if verbose >= 1:
                print(f"Creating nearest neighbor object with {all_zxy.shape[0]} neurons")
                print(f"And test point: {target_pt}")
                if verbose >= 2:
                    candidate_names = [n for i, n in enumerate(all_tracklet_names) if nonnan_ind[i]]
                    print(f"These tracklets were possible: {candidate_names}")
            nbr_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(all_zxy)
        dist, ind_local_coords = nbr_obj.kneighbors([target_pt], n_neighbors=1)
        ind_local_coords = ind_local_coords[0][0]
        if verbose >= 1:
            print(ind_local_coords)
            print(f"Closest point is: {all_zxy[ind_local_coords, :]}")
        ind_global_coords = np.where(nonnan_ind)[0][ind_local_coords]
        tracklet_name = all_tracklet_names[ind_global_coords]

    return dist, ind_global_coords, tracklet_name