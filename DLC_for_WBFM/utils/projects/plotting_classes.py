import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union, Dict
from copy import deepcopy
import numpy as np
import pandas as pd
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors

from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.utils_filepaths import lexigraphically_sort, SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename
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
    global2tracklet: Dict[str, List[str]]
    # df_final_tracks: pd.DataFrame
    segmentation_metadata: DetectedNeurons

    # Annotation
    manual_global2tracklet_names: Dict[str, List[str]] = None
    manual_global2tracklet_removals: Dict[str, List[str]] = None
    current_neuron: str = None
    current_tracklet_name: Union[str, None] = None

    # Saving
    training_cfg: SubfolderConfigFile = None
    tracking_cfg: SubfolderConfigFile = None

    # Visualization options
    refresh_callback: callable = None
    to_add_layer_to_viewer: bool = True
    verbose: int = 1

    def __post_init__(self):
        if self.manual_global2tracklet_names is None:
            self.manual_global2tracklet_names = defaultdict(list)
        if self.manual_global2tracklet_removals is None:
            self.manual_global2tracklet_removals = defaultdict(list)

    @property
    def combined_global2tracklet_dict(self):
        tmp = deepcopy(self.global2tracklet)
        for k in tmp.keys():
            tmp[k].extend(self.manual_global2tracklet_names[k].copy())
            [tmp[k].remove(neuron) for neuron in self.manual_global2tracklet_removals[k]]
        if self.current_tracklet_name is not None:
            logging.warning("Currently active tracklet not included in combined dict")
        return tmp

    def calculate_tracklets_for_neuron(self, neuron_name=None) -> List[pd.DataFrame]:
        # Note: does NOT save this neuron as self.current_neuron
        if neuron_name is None:
            neuron_name = self.current_neuron
        if neuron_name is None:
            raise ValueError("Must pass neuron name explicitly or have one saved in the object")
        # Returns a list of pd.DataFrames with columns x, y, z, and likelihood, which can be plotted in a loop
        these_names = self.global2tracklet[neuron_name].copy()
        # all_tracklet_names = lexigraphically_sort(list(self.df_tracklets.columns.levels[0]))
        # all_tracklet_names = list(self.df_tracklets.columns.levels[0])

        # these_names = [all_tracklet_names[i] for i in tracklet_ind]
        these_names.extend(self.manual_global2tracklet_names[neuron_name])
        if self.current_tracklet_name is not None:
            these_names.append(self.current_tracklet_name)

        if self.verbose >= 1:
            self.print_current_status(neuron_name)
        these_tracklets = [self.df_tracklets[name] for name in these_names]

        return these_tracklets

    def get_neuron_name_of_conflicting_match(self, tracklet_name=None):
        # The tracklet shouldn't be in the manually annotated match, because it can't be added if there are conflicts
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        for k, v in self.combined_global2tracklet_dict.items():
            if tracklet_name in v:
                return k
        return None

    def is_tracklet_already_matched(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        name = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if name is None:
            return False
        else:
            return True

    def get_tracklet_names_of_time_conflicts(self, tracklet_name=None) -> Union[dict, None]:
        # The tracklet shouldn't be in the manually annotated match, because it can't be added if there are conflicts
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        if tracklet_name is None:
            return None
        current_tracklet_names = self.combined_global2tracklet_dict[self.current_neuron]
        df_target_tracklet = self.df_tracklets[tracklet_name]
        times_target_tracklet = df_target_tracklet['z'].dropna().index

        all_overlaps = {}
        for name in current_tracklet_names:
            current_tracklet = self.df_tracklets[name]
            times_current_tracklet = current_tracklet['z'].dropna().index
            intersection = times_target_tracklet.intersection(times_current_tracklet)
            if len(intersection) > 0:
                all_overlaps[name] = intersection

        if len(all_overlaps) == 0:
            return None
        else:
            return all_overlaps

    def tracklet_has_time_overlap(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        name_list = self.get_tracklet_names_of_time_conflicts(tracklet_name)
        if name_list is None:
            return False
        else:
            return True

    @property
    def is_current_tracklet_confict_free(self):
        has_match = self.is_tracklet_already_matched()
        has_time_overlap = self.tracklet_has_time_overlap()
        if not has_match and not has_time_overlap:
            return True
        else:
            self.print_tracklet_conflicts()
            return False

    def print_tracklet_conflicts(self):
        name = self.get_neuron_name_of_conflicting_match()
        if name is not None:
            print(f"Tracklet {self.current_tracklet_name} is already matched to other neuron: {name}")
        name_dict = self.get_tracklet_names_of_time_conflicts()
        if name_dict is not None:
            print(f"Current tracklet {self.current_tracklet_name} has time conflict with tracklet(s):")
            for name, times in name_dict.items():
                print(f"{name} at times {times}")

    def remove_tracklet_from_other_match(self):
        tracklet_name = self.current_tracklet_name
        other_match = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if other_match is not None:
            self.manual_global2tracklet_removals[other_match].append(tracklet_name)
            assert not self.is_tracklet_already_matched(tracklet_name), f"Removal of {tracklet_name} from {other_match} failed"
        else:
            print("Already unmatched")

    def remove_tracklets_with_time_conflicts(self):
        tracklet_name = self.current_tracklet_name
        all_overlap_dict = self.get_tracklet_names_of_time_conflicts(tracklet_name)
        if all_overlap_dict is not None:
            for conflicting_tracklet_name in all_overlap_dict.keys():
                self.manual_global2tracklet_removals[self.current_neuron].append(conflicting_tracklet_name)

            assert not self.tracklet_has_time_overlap(tracklet_name), f"Clean up of {tracklet_name} failed"
        else:
            print("Already not conflicting")

    def save_current_tracklet_to_neuron(self):
        if self.is_current_tracklet_confict_free:

            d = self.manual_global2tracklet_names[self.current_neuron]
            if self.current_tracklet_name in d:
                print(f"Tracklet {self.current_tracklet_name} already in {self.current_neuron}; nothing added")
            else:
                d.append(self.current_tracklet_name)
                print(f"Successfully added tracklet {self.current_tracklet_name} to {self.current_neuron}")
            self.current_tracklet_name = None
        else:
            print("Current tracklet has conflicts, please resolve before saving as a match")

    def print_current_status(self, neuron_name=None):
        if neuron_name is None:
            neuron_name = self.current_neuron

        if neuron_name is None:
            print("No neuron selected")
        else:
            these_names = self.global2tracklet[self.current_neuron]
            print(f"Initial tracklets for {self.current_neuron}: {these_names}")
            print(f"Previous manually added tracklets: {self.manual_global2tracklet_names[neuron_name]}")
            print(f"Previous manually removed tracklets: {self.manual_global2tracklet_removals[neuron_name]}")
            print(f"Currently selected (not yet added) tracklet: {self.current_tracklet_name}")

    def save_manual_matches_to_disk(self):
        # Saves the new dataframe (possibly with split tracklets) and the new matches
        logging.warning("Saving tracklet dataframe, may take a while")

        match_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname')
        # match_fname = get_sequential_filename(match_fname)
        self.tracking_cfg.pickle_in_local_project(self.combined_global2tracklet_dict, match_fname)
        match_fname = self.tracking_cfg.unresolve_absolute_path(match_fname)
        self.tracking_cfg.config.update({'manual_correction_global2tracklet_fname': match_fname})

        df_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname')
        # df_fname = get_sequential_filename(df_fname)
        self.tracking_cfg.h5_in_local_project(self.df_tracklets, df_fname)
        df_fname = self.tracking_cfg.unresolve_absolute_path(df_fname)
        self.tracking_cfg.config.update({'manual_correction_tracklets_df_fname': df_fname})

        # df_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_3d_tracks_df_fname')
        # df_fname = get_sequential_filename(df_fname)
        # self.tracking_cfg.h5_in_local_project(self.df_final_tracks, df_fname)
        # df_fname = self.tracking_cfg.unresolve_absolute_path(df_fname)
        # self.tracking_cfg.config.update({'manual_correction_3d_tracks_df_fname': df_fname})

        logging.info("Saving successful!")
        self.tracking_cfg.update_on_disk()

    def split_current_tracklet(self, i_time):
        # The current time is included in the "new half" of the tracklet
        # The newer half is added as a new index in the df_tracklet dataframe
        # And finally, the newer half is set as the current tracklet
        if self.current_tracklet_name is None:
            print("No current tracklet!")
            return

        old_name = self.current_tracklet_name
        this_tracklet = self.df_tracklets[[old_name]]

        # Split
        old_half = this_tracklet.copy()
        new_half = this_tracklet.copy()

        old_half.iloc[i_time:] = np.nan
        new_half.iloc[:i_time] = np.nan
        new_name = self.get_next_tracklet_name()
        new_half.rename(columns={old_name: new_name}, level=0, inplace=True)

        print(f"Creating new tracklet {new_name} from {old_name} by splitting at t={i_time}")
        print(f"New non-nan lengths: new: {new_half[new_name]['z'].count()}, old:{old_half[old_name]['z'].count()}")

        # Save
        self.df_tracklets = pd.concat([self.df_tracklets, new_half], axis=1)
        self.df_tracklets[old_name] = old_half[old_name]
        self.current_tracklet_name = new_name

    def clear_current_tracklet(self):
        if self.current_tracklet_name is not None:
            print(f"Cleared tracklet {self.current_tracklet_name}")
            self.current_tracklet_name = None

    def get_next_tracklet_name(self):
        all_names = list(self.df_tracklets.columns.levels[0])
        # Really want to make sure we are after all other names,
        i_tracklet = int(1e6 + len(all_names) + 1)
        build_tracklet_name = lambda i: f'neuron{i}'
        new_name = build_tracklet_name(i_tracklet)
        while new_name in all_names:
            i_tracklet += 1
            new_name = build_tracklet_name(i_tracklet)
        return new_name

    def connect_tracklet_clicking_callback(self, layer_to_add_callback, viewer,
                                           max_dist=10.0,
                                           refresh_callback=None):
        self.refresh_callback = refresh_callback

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):
            seg_index = layer.get_value(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True
            )

            if seg_index is None:
                return

            if self.verbose >= 1:
                print(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                      f"and position {event.position[1:]}")
            dist, ind, tracklet_name = self.get_tracklet_from_segmentation_index(
                i_time=int(event.position[0]),
                seg_ind=seg_index
            )
            # dist, ind, tracklet_name = self.get_closest_tracklet_to_point(
            #     i_time=int(event.position[0]),
            #     target_pt=event.position[1:],
            #     verbose=1
            # )

            dist = dist[0][0]
            if self.verbose >= 1:
                print(f"Neuron is part of tracklet {tracklet_name} with distance {dist}")

            if dist < max_dist:
                self.current_tracklet_name = tracklet_name
                if self.current_neuron is not None:
                    # self.manual_global2tracklet_names[self.current_neuron].append(tracklet_name)
                    self.refresh_callback()

                df_single_track = self.df_tracklets[tracklet_name]
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

    def get_closest_tracklet_to_point(self,
                                      i_time,
                                      target_pt,
                                      nbr_obj: NearestNeighbors = None,
                                      nonnan_ind=None,
                                      verbose=0):
        df_tracklets = self.df_tracklets
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

    def get_tracklet_from_segmentation_index(self, i_time, seg_ind):

        # TODO: Directly use the neuron id - tracklet id matching dataframe
        target_pt = self.segmentation_metadata.mask_index_to_zxy(i_time, seg_ind)
        return self.get_closest_tracklet_to_point(i_time, target_pt)