import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Tuple
from copy import deepcopy

import napari
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import get_time_overlap_of_candidate_tracklet, \
    split_tracklet
from DLC_for_WBFM.utils.pipeline.tracklet_class import DetectedTrackletsAndNeurons
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors
from segmentation.util.utils_postprocessing import split_neuron_interactive
from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.utils_filepaths import SubfolderConfigFile, pickle_load_binary, read_if_exists
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

    tspan: list = None

    verbose: int = 1

    def __post_init__(self):
        if self.tspan is None:
            self.tspan = list(range(self.red_traces.shape[0]))

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

    # df_tracklets: pd.DataFrame
    df_tracklet_obj: DetectedTrackletsAndNeurons
    global2tracklet: Dict[str, List[str]]
    # df_final_tracks: pd.DataFrame
    segmentation_metadata: DetectedNeurons

    # Annotation
    manual_global2tracklet_names: Dict[str, List[str]] = None
    manual_global2tracklet_removals: Dict[str, List[str]] = None
    current_neuron: str = None
    current_tracklet_name: Union[str, None] = None

    tracklet_split_names: Dict[str, List[str]] = None
    tracklet_split_times: Dict[str, List[Tuple[int, int]]] = None

    # Saving
    training_cfg: SubfolderConfigFile = None
    tracking_cfg: SubfolderConfigFile = None

    output_match_fname: str = None
    output_df_fname: str = None
    tracklet_split_names_fname: str = None
    tracklet_split_times_fname: str = None

    saving_lock: threading.Lock = threading.Lock()

    # New: for segmentation interactive splitting
    candidate_mask: np.ndarray = None
    time_of_candidate: int = None
    index_of_original_neuron: int = None

    # Visualization options
    refresh_callback: callable = None
    to_add_layer_to_viewer: bool = True
    verbose: int = 1

    def __post_init__(self):
        if self.manual_global2tracklet_names is None:
            self.manual_global2tracklet_names = defaultdict(list)
        if self.manual_global2tracklet_removals is None:
            self.manual_global2tracklet_removals = defaultdict(list)

        match_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname')
        self.output_match_fname = get_sequential_filename(match_fname)
        df_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname')
        self.output_df_fname = get_sequential_filename(df_fname)

        # Read metadata (if found) and save as same name with suffix
        splits_names_fname = Path(df_fname).parent.joinpath("split_names.pickle")
        splits_times_fname = splits_names_fname.with_name("split_times.pickle")

        reader = pickle_load_binary
        self.tracklet_split_names = read_if_exists(splits_names_fname, reader=reader)
        if self.tracklet_split_names is None:
            self.tracklet_split_names = defaultdict(list)
        # else:
        #     # Do not overwrite old splits?
        #     self.tracklet_split_names_fname = get_sequential_filename(str(splits_names_fname))
        self.tracklet_split_times = read_if_exists(splits_times_fname, reader=reader)
        if self.tracklet_split_times is None:
            self.tracklet_split_times = defaultdict(list)

        self.tracklet_split_names_fname = str(splits_names_fname)
        self.tracklet_split_times_fname = str(splits_times_fname)

        # self.tracklet_split_times_fname = get_sequential_filename(str(splits_times_fname))

        print(f"Output files: {match_fname}, {df_fname}, {splits_names_fname}, {splits_times_fname}")

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
        # all_tracklet_names = lexigraphically_sort(list(self.df_tracklet_obj.data.columns.levels[0]))
        # all_tracklet_names = list(self.df_tracklet_obj.data.columns.levels[0])

        # these_names = [all_tracklet_names[i] for i in tracklet_ind]
        these_names.extend(self.manual_global2tracklet_names[neuron_name])
        if self.current_tracklet_name is not None:
            these_names.append(self.current_tracklet_name)

        if self.verbose >= 1:
            self.print_current_status(neuron_name)
        these_tracklets = [self.df_tracklet_obj.df_tracklets_zxy[name] for name in these_names]

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

    def get_dict_of_tracklet_time_conflicts(self, candidate_tracklet_name=None) -> Union[Dict[str, list], None]:
        # The tracklet shouldn't be in the manually annotated match, because it can't be added if there are conflicts
        if candidate_tracklet_name is None:
            candidate_tracklet_name = self.current_tracklet_name
        if candidate_tracklet_name is None:
            return None
        tracklet_dict = self.combined_global2tracklet_dict
        current_tracklet_names = tracklet_dict[self.current_neuron]
        df_tracklets = self.df_tracklet_obj.df_tracklets_zxy

        return get_time_overlap_of_candidate_tracklet(candidate_tracklet_name, current_tracklet_names, df_tracklets)

    def time_of_next_conflict(self, i_start=0) -> Tuple[Union[int, None], Union[str, None]]:
        conflicts = self.get_dict_of_tracklet_time_conflicts()
        if conflicts is None:
            return None, None

        next_conflict_time = np.inf
        neuron_conflict = None
        for neuron, times in conflicts.items():
            t = np.min(times)
            if i_start < t < next_conflict_time:
                next_conflict_time = t
                neuron_conflict = neuron

        if neuron_conflict is not None:
            print(f"Found next conflict with {neuron_conflict} at time {next_conflict_time}")
            return int(next_conflict_time), neuron_conflict
        else:
            return None, None

    def tracklet_has_time_overlap(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        name_list = self.get_dict_of_tracklet_time_conflicts(tracklet_name)
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
        name_dict = self.get_dict_of_tracklet_time_conflicts()
        if name_dict is not None:
            print(f"Current tracklet {self.current_tracklet_name} has time conflict with tracklet(s):")
            for name, times in name_dict.items():
                print(f"{name} at times {times}")

    def remove_tracklet_from_all_matches(self):
        tracklet_name = self.current_tracklet_name
        other_match = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if other_match is not None:
            with self.saving_lock:
                self.manual_global2tracklet_removals[other_match].append(tracklet_name)
            assert not self.is_tracklet_already_matched(tracklet_name), f"Removal of {tracklet_name} from {other_match} failed"
        else:
            print("Already unmatched")

    def remove_tracklets_with_time_conflicts(self):
        tracklet_name = self.current_tracklet_name
        all_overlap_dict = self.get_dict_of_tracklet_time_conflicts(tracklet_name)
        if all_overlap_dict is not None:
            with self.saving_lock:
                for conflicting_tracklet_name in all_overlap_dict.keys():
                    self.manual_global2tracklet_removals[self.current_neuron].append(conflicting_tracklet_name)

            assert not self.tracklet_has_time_overlap(tracklet_name), f"Clean up of {tracklet_name} failed"
        else:
            print("Already not conflicting")

    def save_current_tracklet_to_neuron(self):
        if self.current_tracklet_name is None:
            print("No neuron selected")
            return
        if self.is_current_tracklet_confict_free:

            with self.saving_lock:
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

    def save_manual_matches_to_disk_dispatch(self):
        # Saves the new dataframe (possibly with split tracklets) and the new matches
        logging.warning("Saving tracklet dataframe, DO NOT QUIT")
        print("Note: the GUI will still respond, but you can't split or save any tracklets")
        # self.saving_lock.acquire(blocking=True)
        logging.warning("Acquired saving lock; currently saving")
        t = threading.Thread(target=self.save_manual_matches_to_disk)
        t.start()

    def save_manual_matches_to_disk(self):
        with self.saving_lock:
            self.tracking_cfg.pickle_in_local_project(self.combined_global2tracklet_dict, self.output_match_fname)
            match_fname = self.tracking_cfg.unresolve_absolute_path(self.output_match_fname)
            self.tracking_cfg.config.update({'manual_correction_global2tracklet_fname': match_fname})

            self.tracking_cfg.h5_in_local_project(self.df_tracklet_obj.df_tracklets_zxy, self.output_df_fname)
            df_fname = self.tracking_cfg.unresolve_absolute_path(self.output_df_fname)
            self.tracking_cfg.config.update({'manual_correction_tracklets_df_fname': df_fname})

            self.tracking_cfg.pickle_in_local_project(self.tracklet_split_names, self.tracklet_split_names_fname)
            self.tracking_cfg.pickle_in_local_project(self.tracklet_split_times, self.tracklet_split_times_fname)

            print("Saving successful! You may now quit")
            self.tracking_cfg.update_on_disk()
        # finally:
        #     self.saving_lock.release()

    def split_current_tracklet(self, i_split, set_new_half_to_current=True):
        # The current time is included in the "new half" of the tracklet
        # The newer half is added as a new index in the df_tracklet dataframe
        # And finally, the newer half is set as the current tracklet
        if self.current_tracklet_name is None:
            print("No current tracklet!")
            return

        with self.saving_lock:
            # Left half stays as old name
            old_name = self.current_tracklet_name
            all_tracklets = self.df_tracklet_obj.df_tracklets_zxy

            all_tracklets, left_name, right_name = split_tracklet(all_tracklets, i_split, old_name)

            # Save
            # self.df_tracklet_obj.data = pd.concat([self.df_tracklet_obj.data, new_half], axis=1)
            # self.df_tracklet_obj.data[old_name] = old_half[old_name]

            self.df_tracklet_obj.df_tracklets_zxy = all_tracklets
            if set_new_half_to_current:
                self.current_tracklet_name = right_name
            else:
                self.current_tracklet_name = left_name

            # Save a record of the split
            self.tracklet_split_names[left_name].append(right_name)
            self.tracklet_split_times[left_name].append((i_split - 1, i_split))

    def clear_current_tracklet(self):
        if self.current_tracklet_name is not None:
            print(f"Cleared tracklet {self.current_tracklet_name}")
            self.current_tracklet_name = None
        else:
            print("No current tracklet; this button did nothing")

    def connect_tracklet_clicking_callback(self, layer_to_add_callback, viewer: napari.Viewer,
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
            time_index = int(event.position[0])
            # print("Event modifiers")
            # print(event.modifiers)
            # The modifiers field is a list of Key objects
            # Class definition: https://github.com/vispy/vispy/blob/ef982591e223fff09d91d8c2697489c7193a85aa/vispy/util/keys.py
            # print([m.name for m in event.modifiers])
            if 'Alt' in [m.name for m in event.modifiers]:
                logging.info("Unset Segmentation-click interaction triggered (modifier=alt)")
                # TODO: Add different function
                # TODO: user-defined kwargs
                full_mask = viewer.layers['Raw segmentation'].data[time_index]
                red_volume = viewer.layers['Red data'].data[time_index]
                new_full_mask = split_neuron_interactive(full_mask, red_volume, seg_index,
                                                         min_separation=2,
                                                         which_neuron_keeps_original='top',
                                                         verbose=3)

                if new_full_mask is None:
                    return
                # Add as a new candidate layer
                layer_name = f"Candidate_split_of_n{seg_index}_at_t{time_index}"
                viewer.add_labels(new_full_mask, name=layer_name, opacity=1.0)

                # Save for later combining with original mask
                self.candidate_mask = new_full_mask
                self.time_of_candidate = time_index
                self.index_of_original_neuron = seg_index

                return
            else:
                logging.info("Tracklet segmentation-click interaction triggered")

            if seg_index is None or seg_index == 0:
                print("Event triggered on background; returning")
                return

            if self.verbose >= 1:
                print(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                      f"and position {event.position[1:]}")
            dist, ind, tracklet_name = self.df_tracklet_obj.get_tracklet_from_segmentation_index(
                i_time=time_index,
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

                df_single_track = self.df_tracklet_obj.df_tracklets_zxy[tracklet_name]
                if self.verbose >= 1:
                    print(f"Adding tracklet of length {df_single_track['z'].count()}")
                if self.to_add_layer_to_viewer:
                    all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track)
                    viewer.add_tracks(track_of_point, name=tracklet_name)

                if self.verbose >= 2:
                    print(df_single_track.dropna(inplace=False))
            else:
                if self.verbose >= 1:
                    print(f"WARNING: Tracklet too far away; not adding anything")
