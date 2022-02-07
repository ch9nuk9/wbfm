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

from DLC_for_WBFM.utils.tracklets.utils_tracklets import get_time_overlap_of_candidate_tracklet, \
    split_tracklet_within_dataframe
from DLC_for_WBFM.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons
from segmentation.util.utils_metadata import DetectedNeurons
from segmentation.util.utils_postprocessing import split_neuron_interactive
from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.project_config_classes import SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_filenames import read_if_exists, pickle_load_binary
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
class TrackletAndSegmentationAnnotator:

    df_tracklet_obj: DetectedTrackletsAndNeurons
    global2tracklet: Dict[str, List[str]]
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

    # New: for interactive segmentation splitting and merging
    candidate_mask: np.ndarray = None
    time_of_candidate: int = None
    indices_of_original_neurons: List[int] = None

    segmentation_options: dict = None

    # Visualization options
    segmentation_callbacks: List[callable] = None
    tracklet_callbacks: List[callable] = None
    to_add_layer_to_viewer: bool = True
    verbose: int = 1

    is_currently_interactive: bool = True

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

        if self.segmentation_options is None:
            self.segmentation_options = dict(
                x_split_local_coord=None
            )

        if self.indices_of_original_neurons is None:
            self.indices_of_original_neurons = []

        print(f"Output files for annotator: {match_fname}, {df_fname}, {splits_names_fname}, {splits_times_fname}")

    @property
    def combined_global2tracklet_dict(self):
        tmp = deepcopy(self.global2tracklet)
        for k in tmp.keys():
            tmp[k].extend(self.manual_global2tracklet_names[k].copy())
            [tmp[k].remove(neuron) for neuron in self.manual_global2tracklet_removals[k] if neuron in tmp[k]]
        if self.current_tracklet_name is not None:
            pass
            # logging.warning("Currently active tracklet not included in combined dict")
        return tmp

    @property
    def current_tracklet(self):
        if self.current_tracklet_name is None:
            return None
        df_single_track = self.df_tracklet_obj.df_tracklets_zxy[self.current_tracklet_name]
        return df_single_track

    def set_current_tracklet(self, tracklet_name):
        self.current_tracklet_name = tracklet_name

    def clear_current_tracklet(self):
        if self.current_tracklet_name is not None:
            print(f"Cleared tracklet {self.current_tracklet_name}")
            self.current_tracklet_name = None
        else:
            print("No current tracklet; this button did nothing")

    def calculate_tracklets_for_neuron(self, neuron_name=None) -> List[pd.DataFrame]:
        # Note: does NOT save this neuron as self.current_neuron
        if neuron_name is None:
            neuron_name = self.current_neuron
        if neuron_name is None:
            raise ValueError("Must pass neuron name explicitly or have one saved in the object")
        # Returns a list of pd.DataFrames with columns x, y, z, and likelihood, which can be plotted in a loop
        these_names = self.global2tracklet[neuron_name].copy()

        these_names.extend(self.manual_global2tracklet_names[neuron_name])
        [these_names.remove(name) for name in self.manual_global2tracklet_removals[neuron_name]]
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

    def end_time_of_current_tracklet(self):
        tracklet = self.current_tracklet
        if tracklet is None:
            return None
        else:
            return int(tracklet.last_valid_index())

    def start_time_of_current_tracklet(self):
        tracklet = self.current_tracklet
        if tracklet is None:
            return None
        else:
            return int(tracklet.first_valid_index())

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

    def get_types_of_conflicts(self):
        types_of_conflicts = []
        conflicting_match = self.get_neuron_name_of_conflicting_match()
        if conflicting_match and conflicting_match == self.current_neuron:
            types_of_conflicts.append("Already added")
        else:
            if conflicting_match:
                types_of_conflicts.append("Identity")
            if self.get_dict_of_tracklet_time_conflicts():
                types_of_conflicts.append("Time")
            if len(types_of_conflicts) == 0:
                types_of_conflicts.append("No conflicts")
        return types_of_conflicts

    def print_tracklet_conflicts(self):
        name = self.get_neuron_name_of_conflicting_match()
        if name is not None:
            print(f"Tracklet {self.current_tracklet_name} is already matched to other neuron: {name}")
        name_dict = self.get_dict_of_tracklet_time_conflicts()
        if name_dict is not None:
            print(f"Current tracklet {self.current_tracklet_name} has time conflict with tracklet(s):")
            for name, times in name_dict.items():
                print(f"{name} at times {times}")

    def add_tracklet_to_neuron(self, tracklet_name, neuron_name):
        previously_added = self.manual_global2tracklet_names[neuron_name]
        previously_removed = self.manual_global2tracklet_removals[neuron_name]
        if tracklet_name in previously_added:
            print(f"Tracklet {tracklet_name} already in {neuron_name}; nothing added")
        else:
            self.manual_global2tracklet_names[neuron_name].append(tracklet_name)
            print(f"Successfully added tracklet {tracklet_name} to {neuron_name}")

        if tracklet_name in previously_removed:
            print(f"Tracklet was in the to-remove list, but was removed")
            self.manual_global2tracklet_removals[neuron_name].remove(tracklet_name)

    def remove_tracklet_from_neuron(self, tracklet_name, neuron_name):
        previously_added = self.manual_global2tracklet_names[neuron_name]
        previously_removed = self.manual_global2tracklet_removals[neuron_name]
        if tracklet_name in previously_removed:
            print(f"Tracklet {tracklet_name} already removed from {neuron_name}; nothing removed")
        else:
            self.manual_global2tracklet_removals[neuron_name].append(tracklet_name)
            print(f"Successfully added {tracklet_name} to removal list of {neuron_name}")

        if tracklet_name in previously_added:
            print(f"{tracklet_name} was in the manually to-add list, but was removed")
            self.manual_global2tracklet_names[neuron_name].remove(tracklet_name)

    def remove_tracklet_from_all_matches(self):
        tracklet_name = self.current_tracklet_name
        other_match = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if other_match is not None:
            with self.saving_lock:
                self.remove_tracklet_from_neuron(tracklet_name, other_match)
            # assert not self.is_tracklet_already_matched(tracklet_name), f"Removal of {tracklet_name} from {other_match} failed"
        else:
            print("Already unmatched")

    def remove_tracklets_with_time_conflicts(self):
        tracklet_name = self.current_tracklet_name
        all_overlap_dict = self.get_dict_of_tracklet_time_conflicts(tracklet_name)
        if all_overlap_dict is not None:
            with self.saving_lock:
                for conflicting_tracklet_name in all_overlap_dict.keys():
                    self.remove_tracklet_from_neuron(conflicting_tracklet_name, self.current_neuron)
                    # self.manual_global2tracklet_removals[self.current_neuron].append(conflicting_tracklet_name)
            # assert not self.tracklet_has_time_overlap(tracklet_name), f"Clean up of {tracklet_name} failed"
        else:
            print("Already not conflicting")

    def save_current_tracklet_to_current_neuron(self):
        if self.current_tracklet_name is None:
            print("No neuron selected")
            return None
        if self.is_current_tracklet_confict_free:
            with self.saving_lock:
                self.add_tracklet_to_neuron(self.current_tracklet_name, self.current_neuron)
                tracklet_name = self.current_tracklet_name
                self.current_tracklet_name = None
            return tracklet_name
        else:
            print("Current tracklet has conflicts, please resolve before saving as a match")
            return None

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

            all_tracklets, left_name, right_name = split_tracklet_within_dataframe(all_tracklets, i_split, old_name)

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

    def segmentation_updated_callbacks(self):
        [callback() for callback in self.segmentation_callbacks]

    def tracklet_updated_callbacks(self):
        [callback() for callback in self.tracklet_callbacks]

    def connect_tracklet_clicking_callback(self, layer_to_add_callback, viewer: napari.Viewer,
                                           added_segmentation_callbacks,
                                           added_tracklet_callbacks,
                                           max_dist=1.0):
        self.segmentation_callbacks = added_segmentation_callbacks
        self.tracklet_callbacks = added_tracklet_callbacks

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):

            if not self.is_currently_interactive:
                logging.warning("Click received, but interactivity is turned off")
                return

            # Get information about clicked-on neuron
            seg_index = layer.get_value(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True
            )
            time_index = int(event.position[0])
            if seg_index is None or seg_index == 0:
                print("Event triggered on background; returning")
                return

            if self.verbose >= 1:
                print(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                      f"and position {event.position[1:]}")

            # Decide which mode: segmentation or tracklet

            # The modifiers field is a list of Key objects
            # Class definition: https://github.com/vispy/vispy/blob/ef982591e223fff09d91d8c2697489c7193a85aa/vispy/util/keys.py
            # print([m.name for m in event.modifiers])
            click_modifiers = [m.name.lower() for m in event.modifiers]
            if 'control' in click_modifiers:
                # Just add neuron, no automatic splitting
                self.append_segmentation_to_list(time_index, seg_index)
                segment_mode_not_tracklet_mode = True
            elif 'alt' in click_modifiers:
                # Shortcut for clearing all neurons, adding this one, and attempting to split
                self.set_selected_segmentation(time_index, seg_index)
                self.split_current_neuron_and_add_napari_layer(viewer, split_method="Gaussian")
                segment_mode_not_tracklet_mode = True
            else:
                segment_mode_not_tracklet_mode = False

            if segment_mode_not_tracklet_mode:
                return

            # Split tracklet, not segmentation
            tracklet_name = self.df_tracklet_obj.get_tracklet_from_segmentation_index(
                i_time=time_index,
                seg_ind=seg_index
            )
            # dist, ind, tracklet_name = self.df_tracklet_obj.get_tracklet_from_segmentation_index(
            #     i_time=time_index,
            #     seg_ind=seg_index
            # )

            # dist = dist[0][0]
            if self.verbose >= 1:
                print(f"Neuron is part of tracklet {tracklet_name}")

            # if dist < max_dist:
            if tracklet_name:
                self.set_current_tracklet(tracklet_name)
                self.add_current_tracklet_to_viewer(viewer)
                if self.current_neuron is not None:
                    self.tracklet_updated_callbacks()
            else:
                self.set_selected_segmentation(time_index, seg_index)
                if self.verbose >= 1:
                    print(f"Tracklet not found; adding segmentation only")

    def add_current_tracklet_to_viewer(self, viewer):
        df_single_track = self.current_tracklet
        if self.verbose >= 1:
            print(f"Adding tracklet of length {df_single_track['z'].count()}")
        if self.to_add_layer_to_viewer:
            all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track)
            viewer.add_tracks(track_of_point, name=self.current_tracklet_name,
                              tail_width=10, head_length=1, tail_length=4)
        if self.verbose >= 2:
            print(df_single_track.dropna(inplace=False))

    def split_current_neuron_and_add_napari_layer(self, viewer, split_method):
        seg_index = self.indices_of_original_neurons
        if len(seg_index) > 1:
            print("Multiple neurons selected, splitting is ambiguous... returning")
            return
        else:
            seg_index = seg_index[0]
            time_index = self.time_of_candidate

        full_mask = viewer.layers['Raw segmentation'].data[time_index]
        red_volume = viewer.layers['Red data'].data[time_index]
        new_full_mask = split_neuron_interactive(full_mask, red_volume, seg_index,
                                                 min_separation=2,
                                                 verbose=3,
                                                 method=split_method,
                                                 **self.segmentation_options)
        split_succeeded = new_full_mask is not None
        if split_succeeded:
            # Add as a new candidate layer
            layer_name = f"Candidate_split_at_t{time_index}"
            viewer.add_labels(new_full_mask, name=layer_name, opacity=1.0)

            # Save for later combining with original mask
            self.candidate_mask = new_full_mask

    def merge_current_neurons(self, viewer):
        # NOTE: will keep the index of the first selected neuron
        if len(self.indices_of_original_neurons) <= 1:
            print(f"Too few neurons selected ({len(self.indices_of_original_neurons)}), aborting")
            return
        elif len(self.indices_of_original_neurons) > 2:
            print(f"Merging than 2 neurons not supported, aborting")
            return

        time_index = self.time_of_candidate
        new_full_mask = viewer.layers['Raw segmentation'].data[time_index].copy()

        indices_to_overwrite = self.indices_of_original_neurons[1:]
        target_index = self.indices_of_original_neurons[0]
        for i in indices_to_overwrite:
            new_full_mask[new_full_mask == i] = target_index

        # Add as a new candidate layer
        layer_name = f"Candidate_split_at_t{time_index}"
        viewer.add_labels(new_full_mask, name=layer_name, opacity=1.0)

        # Save for later combining with original mask
        self.candidate_mask = new_full_mask

        # Update the saved indices to just be the new one
        self.set_selected_segmentation(self.time_of_candidate, target_index)

    def clear_currently_selected_segmentations(self, do_callbacks=True):
        self.time_of_candidate = None
        self.indices_of_original_neurons = []
        if do_callbacks:
            self.segmentation_updated_callbacks()

    def append_segmentation_to_list(self, time_index, seg_index):
        if self.time_of_candidate is None:
            self.time_of_candidate = time_index
            self.indices_of_original_neurons = [seg_index]
            self.segmentation_updated_callbacks()
        else:
            if self.time_of_candidate == time_index:
                self.indices_of_original_neurons.append(seg_index)
                self.segmentation_updated_callbacks()
                print(f"Added neuron to list; current neurons: {self.indices_of_original_neurons}")
            else:
                logging.warning("Attempt to add segmentations of different time points; not supported")

    def set_selected_segmentation(self, time_index, seg_index):
        """Like append_segmentation_to_list, but forces a single neuron. Also properly accounts for callbacks"""
        self.clear_currently_selected_segmentations(do_callbacks=False)
        self.append_segmentation_to_list(time_index, seg_index)

    def toggle_highlight_selected_neuron(self, viewer):
        layer = viewer.layers['Raw segmentation']
        ind = self.indices_of_original_neurons
        if len(ind) > 1:
            logging.warning("Selection not implemented if more than one neuron is selected")
            return
        elif len(ind) == 1:
            layer.show_selected_label = True
            layer.selected_label = ind[0]
        else:
            layer.show_selected_label = False

    def attach_current_segmentation_to_current_tracklet(self):
        if len(self.indices_of_original_neurons) != 1:
            logging.warning("Can't attach multiple segmentations at once")
            return False

        name = self.current_tracklet_name
        if name is None:
            logging.warning("No tracklet selected; can't attach segmentation")
            return False
        # Get known data, then rebuild the other metadata from this
        t = self.time_of_candidate
        mask_ind = self.indices_of_original_neurons[0]
        segmentation_metadata = self.segmentation_metadata
        df_tracklet_obj = self.df_tracklet_obj

        with self.saving_lock:
            self.df_tracklet_obj.update_tracklet_metadata_using_segmentation_metadata(
                t, name, mask_ind=mask_ind, likelihood=1.0
            )

        self.clear_currently_selected_segmentations(do_callbacks=False)
        self.segmentation_updated_callbacks()
        self.tracklet_updated_callbacks()

        return True
