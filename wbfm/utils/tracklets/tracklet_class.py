import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import sklearn

from wbfm.utils.tracklets.high_performance_pandas import insert_value_in_sparse_df, PaddedDataFrame, \
    get_names_from_df
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import dataframe_to_dataframe_zxy_format, \
    get_names_of_conflicting_dataframes, get_times_of_conflicting_dataframes, get_column_name_from_time_and_column_value
from wbfm.utils.general.custom_errors import AnalysisOutOfOrderError, DataSynchronizationError
from wbfm.utils.neuron_matching.matches_class import MatchesAsGraph, MatchesWithConfidence
from wbfm.utils.projects.utils_filenames import lexigraphically_sort
from wbfm.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.tracklets.training_data_from_tracklets import translate_training_names_to_raw_names
from wbfm.utils.tracklets.utils_tracklets import get_next_name_tracklet_or_neuron


@dataclass
class NeuronComposedOfTracklets:

    name: str = None
    initialization_frame: int = None
    initialization_point: np.ndarray = None  # Only needed if initialized without a tracklet
    # initialization_neuron_name: str

    neuron2tracklets: MatchesAsGraph = None
    tracklet_covering_ind: list = None

    # For detecting outliers in candidate additional tracklet matches
    classifier: sklearn.svm._classes.OneClassSVM = None
    scaler: sklearn.preprocessing._data.StandardScaler = None
    classifier_rejection_threshold: float = 0.0
    fields_to_classify: list = None
    # training_data: np.ndarray = None

    verbose: int = 0

    @property
    def neuron_ind(self):
        return name2int_neuron_and_tracklet(self.name) - 1

    @property
    def name_in_graph(self):
        return self.neuron2tracklets.raw_name_to_network_name(self.name)

    def __post_init__(self):
        if self.tracklet_covering_ind is None:
            self.tracklet_covering_ind = []
        if self.neuron2tracklets is None:
            self.neuron2tracklets = MatchesAsGraph(offset_convention=[True, False],
                                                   naming_convention=['neuron', 'tracklet'],
                                                   name_prefixes=['frame', 'trackletGroup'])
        if self.fields_to_classify is None:
            self.fields_to_classify = ['z', 'volume']
            # fields_to_classify = ['z', 'volume', 'brightness_red'] # Is red stable enough?

    # For use when assigning matches and iterating over time
    @property
    def next_gap(self):
        return self.tracklet_covering_ind[-1] + 1

    def add_tracklet(self, confidence, tracklet: pd.DataFrame, metadata=None,
                     check_using_classifier=False, verbose=0):
        assert np.isfinite(confidence), f"A finite confidence value must be passed, not {confidence}"
        tracklet_name = tracklet.columns.get_level_values(0).drop_duplicates()[0]
        i_tracklet = name2int_neuron_and_tracklet(tracklet_name)
        passed_classifier = True
        if check_using_classifier:
            if self.classifier:
                passed_classifier, _ = self.check_new_tracklet_using_classifier(tracklet[tracklet_name].dropna())
            else:
                logging.warning("Classifier requested but not initialized")
        if not passed_classifier:
            if verbose >= 2:
                print(f"{tracklet_name} did not pass classifier check")
            return False

        match_with_names = [self.name, tracklet_name, confidence]
        is_match_added = self.neuron2tracklets.add_match_if_not_present(match_with_names,
                                                                        node0_metadata=self.name,
                                                                        node1_metadata=tracklet_name,
                                                                        edge_metadata=metadata,
                                                                        convert_ind_to_names=False)
        if is_match_added:
            tracklet_covering = np.where(tracklet[tracklet_name]['z'].notnull())[0]
            self.tracklet_covering_ind.extend(tracklet_covering)
            # self.next_gap = tracklet_covering[-1] + 1

            if self.verbose >= 2:
                print(f"Added tracklet {i_tracklet} to neuron {self.name} with next gap: {self.next_gap}")

        return is_match_added

    def initialize_tracklet_classifier(self, list_of_tracklets: list,
                                       augment_to_minimum_points=200, augmentation_factor=0.2):
        """
        This object doesn't see the raw tracklet data, so it must be sent in the call

        Note that I don't want this classifier to be too harsh, especially if there is only a small amount of initial
        training data (i.e. a short tracklet)
        """
        if len(list_of_tracklets) > 0:
            x = [tracklet[self.fields_to_classify].dropna().to_numpy() for tracklet in list_of_tracklets]
            if len(x) > 1:
                x = np.vstack(x)
            else:
                x = x[0]
        else:
            x = self.initialization_point

        if x is None:
            raise AnalysisOutOfOrderError("self.add_tracklet")

        x0 = x.copy()
        while x.shape[0] < augment_to_minimum_points:
            x_augmented = x0 * (1 + augmentation_factor*np.random.randn(x0.shape[0], x0.shape[1]))
            x = np.vstack([x, x_augmented.copy()])

        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        # self.training_data = x
        self.classifier = OneClassSVM(nu=0.05, gamma=0.05, kernel='rbf').fit(x)

    def check_new_tracklet_using_classifier(self, candidate_tracklet: pd.DataFrame):
        y = candidate_tracklet[self.fields_to_classify]
        try:
            y = self.scaler.transform(y.values)
        except ValueError:
            # Empty tracklet
            logging.warning(f"Attempted to invalid tracklet: {candidate_tracklet}")
            return False, 1.0
        predictions = self.classifier.predict(y)  # -1 means outlier
        fraction_outliers = np.mean(predictions)
        if fraction_outliers < self.classifier_rejection_threshold:
            return False, fraction_outliers
        else:
            return True, fraction_outliers

    def get_raw_tracklet_names(self, minimum_confidence=None):
        """Proper null value is []"""
        network_names = self.get_network_tracklet_names(minimum_confidence=minimum_confidence)
        nodes = self.neuron2tracklets.nodes()
        try:
            tracklet_names = [nodes[n]['metadata'] for n in network_names]
            return tracklet_names
        except TypeError:
            return []

    def get_network_tracklet_names(self, minimum_confidence=None):
        network_names = self.neuron2tracklets.get_all_matches(name=self.name_in_graph)
        if minimum_confidence is not None:
            all_conf = self.get_confidences_of_tracklets([network_names])[0]
            network_names = [n for n, c in zip(network_names, all_conf) if c > minimum_confidence]
        return network_names

    def network2raw_name_tracklet_dict(self, minimum_confidence=None):
        network_names = self.get_network_tracklet_names(minimum_confidence=minimum_confidence)
        nodes = self.neuron2tracklets.nodes()
        try:
            tracklet_names = {n: nodes[n]['metadata'] for n in network_names}
            return tracklet_names
        except TypeError:
            return []

    def get_confidences_of_tracklets(self, list_of_lists_of_tracklet_names):
        overlapping_confidences = []
        for these_names in list_of_lists_of_tracklet_names:
            overlapping_confidences.append(
                [self.neuron2tracklets.get_edge_data(self.name_in_graph, t)['weight'] for t in these_names])
        return overlapping_confidences

    def plot_classifier_boundary(self):
        """Assumes z and volume are the classifier coordinates"""

        # xx, yy = np.meshgrid(np.linspace(0, 30, 1500), np.linspace(100, 1000, 1500))
        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        Z = self.classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Send back to original data space
        xy = self.scaler.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        xx, yy = xy[:, 0], xy[:, 1]
        xx = xx.reshape(Z.shape)
        yy = yy.reshape(Z.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")

    @property
    def num_matched_tracklets(self):
        return len(self.neuron2tracklets) - 1

    def __repr__(self):
        return f"Neuron {self.name} (index={self.neuron_ind}) with {self.num_matched_tracklets} tracklets " \
               f"from time {self.initialization_frame} to {self.next_gap}"

    def pretty_print_matches(self):
        self.neuron2tracklets.pretty_print_single_match(self.name)


@dataclass
class DetectedTrackletsAndNeurons:

    df_tracklets_zxy: pd.DataFrame
    segmentation_metadata: Optional[DetectedNeurons] = None

    # If things are modified, this flag should be set
    dataframe_is_synced_to_disk: bool = True
    dataframe_output_filename: Optional[str] = None

    # Pre-construct this for MUCH faster clicking callbacks
    segmentation_id_to_tracklet_name_database: dict = None
    interactive_mode: bool = False

    # EXPERIMENTAL (but tested)
    use_custom_padded_dataframe: bool = False

    def __post_init__(self):
        self.segmentation_id_to_tracklet_name_database = defaultdict(set)
        if self.interactive_mode:
            self.setup_interactivity()

        if self.segmentation_metadata is None:
            logging.warning("No segmentation metadata provided, the following functions will not work: "
                            "get_neurons_at_time, get_number_of_neurons_at_time"
                            "update_tracklet_metadata_using_segmentation_metadata"
                            "initialize_neurons_at_time in the TrackedWorm class")

        if self.use_custom_padded_dataframe:
            raise NotImplementedError
            # print("Using experimental custom padded dataframe")
            # self.df_tracklets_zxy = PaddedDataFrame.construct_from_basic_dataframe(self.df_tracklets_zxy,
            #                                                                        name_mode='tracklet',
            #                                                                        initial_empty_cols=10000)
        else:
            pass
            # print("Using basic dataframe")

    def setup_interactivity(self):
        subcolumn_to_check = 'raw_segmentation_id'
        names = get_names_from_df(self.df_tracklets_zxy)
        logging.info("Prebuilding clickback dictionary (segmentation to tracklet), will take ~ 1 minute")
        for n in tqdm(names):
            self.update_callback_dictionary_for_single_tracklet(n, subcolumn_to_check)

        self.interactive_mode = True

    def update_callback_dictionary_for_single_tracklet(self, n, subcolumn_to_check='raw_segmentation_id'):
        # Note: doesn't remove matches that may be out of date; those will have to removed by updating another tracklet
        col = self.df_tracklets_zxy[n][subcolumn_to_check].dropna(axis=0)
        idx = col.index
        for t, value in zip(idx, col):
            self.segmentation_id_to_tracklet_name_database[(t, int(value))] = {n}

    @property
    def all_tracklet_names(self):
        return get_names_from_df(self.df_tracklets_zxy)
        # return self.df_tracklets_zxy.columns.get_level_values(0).drop_duplicates()

    def int_to_tracklet_name(self, i):
        return self.all_tracklet_names[i]

    def get_closest_tracklet_to_point(self,
                                      i_time,
                                      target_pt,
                                      nbr_obj: NearestNeighbors = None,
                                      nonnan_ind=None,
                                      verbose=0):
        df_tracklets = dataframe_to_dataframe_zxy_format(self.df_tracklets_zxy)
        all_tracklet_names = lexigraphically_sort(get_names_from_df(df_tracklets))

        if any(np.isnan(target_pt)):
            dist, ind_global_coords, tracklet_name = np.inf, None, None
        else:
            if nbr_obj is None:
                all_zxy = np.reshape(df_tracklets.iloc[i_time, :].to_numpy(), (-1, 3))
                nonnan_ind = ~np.isnan(all_zxy).any(axis=1)
                all_zxy = all_zxy[nonnan_ind]
                if verbose >= 1:
                    print(f"Creating nearest neighbor object with {all_zxy.shape[0]} neurons")
                    print(f"And test point: {target_pt}")
                    if verbose >= 2:
                        candidate_names = [n for i, n in enumerate(all_tracklet_names) if nonnan_ind[i]]
                        print(f"These tracklets were possible: {candidate_names}")
                nbr_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(all_zxy)
            else:
                all_zxy = None
            dist, ind_local_coords = nbr_obj.kneighbors([target_pt], n_neighbors=1)
            ind_local_coords = ind_local_coords[0][0]
            if verbose >= 1 and all_zxy is not None:
                print(ind_local_coords)
                print(f"Closest point is: {all_zxy[ind_local_coords, :]}")
            ind_global_coords = np.where(nonnan_ind)[0][ind_local_coords]
            tracklet_name = all_tracklet_names[ind_global_coords]

        return dist, ind_global_coords, tracklet_name

    def get_tracklet_from_segmentation_index(self, i_time, seg_ind):
        if not self.interactive_mode:
            logging.warning("Interactive mode is not set; this method will always return None")
        names = list(self.segmentation_id_to_tracklet_name_database[(i_time, seg_ind)])

        # NOTE: just uses a different column from get_tracklet_from_neuron_and_time()
        # names = find_top_level_name_by_single_column_entry(self.df_tracklets_zxy, i_time, seg_ind,
        #                                                    subcolumn_to_check='raw_segmentation_id')
        if len(names) == 1:
            return names[0]
        elif len(names) > 1:
            logging.warning(f"Multiple matches found ({names}); taking first one")
            return names[0]
        else:
            return None

    def get_tracklet_from_neuron_and_time(self, i_local_neuron, i_time):
        # NOTE: just uses a different column from get_tracklet_from_segmentation_index()
        df = self.df_tracklets_zxy
        ind, val = get_column_name_from_time_and_column_value(df, i_time, i_local_neuron, 'raw_neuron_ind_in_list')
        return ind, val

    def get_number_of_neurons_at_time(self, t: int):
        return len(self.get_neurons_at_time(t))

    def get_neurons_at_time(self, t: int):
        return self.segmentation_metadata.detect_neurons_from_file(t)

    def update_tracklet_metadata_using_segmentation_metadata(self, t: int,
                                                             tracklet_name: str = None,
                                                             mask_ind: int = None,
                                                             likelihood: float = 1.0,
                                                             verbose=0):
        """
        Allows generation of metadata for a single tracklet

        Either the tracklet name or the mask index must be specified
        If only one is specified, it must give a valid unique value for the other upon database search
        """
        segmentation_metadata = self.segmentation_metadata
        mask_ind, tracklet_name = self.get_mask_or_tracklet_from_other(mask_ind, t, tracklet_name)

        # TODO: check that this doesn't produce a time gap in the tracklet
        row_data, column_names = segmentation_metadata.get_all_metadata_for_single_time(mask_ind, t,
                                                                                        likelihood=likelihood)
        if row_data is None:
            if verbose >= 1:
                print(f"{tracklet_name} previously on segmentation {mask_ind} no longer exists, "
                      f"and was removed at that time point")
            self.delete_data_from_tracklet_at_time(t, tracklet_name)
        else:
            if verbose >= 1:
                print(f"{tracklet_name} on segmentation {mask_ind} at t={t} updated with data: {row_data}")
            idx = pd.MultiIndex.from_product([[tracklet_name], column_names])
            self.df_tracklets_zxy = insert_value_in_sparse_df(self.df_tracklets_zxy, t, idx, row_data)
            self.update_callback_dictionary_for_single_tracklet(tracklet_name)
            self.dataframe_is_synced_to_disk = False

    def delete_data_from_tracklet_at_time(self, t, tracklet_name):
        old_ind = int(self.df_tracklets_zxy.loc[t, (tracklet_name, 'raw_segmentation_id')])
        if tracklet_name in self.segmentation_id_to_tracklet_name_database[(t, old_ind)]:
            self.segmentation_id_to_tracklet_name_database[(t, old_ind)].remove(tracklet_name)

        self.df_tracklets_zxy = insert_value_in_sparse_df(self.df_tracklets_zxy, t, tracklet_name, np.nan)
        # self.update_callback_dictionary_for_single_tracklet(tracklet_name)

    def get_mask_or_tracklet_from_other(self, mask_ind, t, tracklet_name):
        if mask_ind is None and tracklet_name is not None:
            mask_ind = int(self.df_tracklets_zxy.loc[t, (tracklet_name, 'raw_segmentation_id')])
        if tracklet_name is None and mask_ind is not None:
            tracklet_name = self.get_tracklet_from_segmentation_index(t, mask_ind)
        if tracklet_name is None or mask_ind is None:
            logging.warning(f"An input was None, which will cause problems: {mask_ind}, {tracklet_name} (t={t})")
            raise DataSynchronizationError('tracklet_name', 'mask_ind')
        return mask_ind, tracklet_name

    def generate_empty_tracklet_with_correct_format(self):
        new_name = get_next_name_tracklet_or_neuron(self.df_tracklets_zxy)
        current_names = get_names_from_df(self.df_tracklets_zxy)
        name_tmp = current_names[0]
        new_tracklet = self.df_tracklets_zxy[[name_tmp]].copy().rename({name_tmp: new_name}, axis=1)
        new_tracklet[:] = np.nan

        return new_tracklet, new_name

    def initialize_new_empty_tracklet(self):
        new_tracklet, new_name = self.generate_empty_tracklet_with_correct_format()
        self.df_tracklets_zxy = pd.concat([self.df_tracklets_zxy, new_tracklet], axis=1)
        self.dataframe_is_synced_to_disk = False
        return new_name

    def synchronize_dataframe_to_disk(self, force_write=False):
        if self.dataframe_is_synced_to_disk and not force_write:
            logging.info("DataFrame is already synchronized")
        else:
            if self.dataframe_output_filename is not None:
                logging.info(f"Saving at: {self.dataframe_output_filename}")
                # try:
                #     self.df_tracklets_zxy.to_hdf(self.dataframe_output_filename, key="df_with_missing")
                # except TypeError:
                # Do not allow h5 format
                self.df_tracklets_zxy.to_pickle(self.dataframe_output_filename)
                self.dataframe_is_synced_to_disk = True
            else:
                logging.warning("Dataframe syncing attempted, but no filename saved")

    @property
    def num_total_tracklets(self):
        return len(self.all_tracklet_names)

    def __repr__(self):
        return f"DetectedTrackletsAndNeurons object with {self.num_total_tracklets} tracklets"


@dataclass
class TrackedWorm:

    global_name_to_neuron: Dict[str, NeuronComposedOfTracklets] = None
    global_name_to_neuron_backup: Dict[str, NeuronComposedOfTracklets] = None
    detections: DetectedTrackletsAndNeurons = None

    logger: Optional[logging.Logger] = None

    verbose: int = 0

    @property
    def num_neurons(self):
        if self.global_name_to_neuron is None:
            return 0
        else:
            return len(self.global_name_to_neuron)

    @property
    def neuron_names(self):
        if self.global_name_to_neuron is None:
            return []
        else:
            return list(self.global_name_to_neuron.keys())

    def get_next_neuron_name(self):
        return int2name_neuron(self.num_neurons + 1)

    def __post_init__(self):
        if self.global_name_to_neuron is None:
            self.global_name_to_neuron = {}

    def initialize_new_neuron(self, name=None, initialization_frame=0):
        if name is None:
            name = self.get_next_neuron_name()

        new_neuron = NeuronComposedOfTracklets(name, initialization_frame, verbose=self.verbose - 1)
        self.global_name_to_neuron[name] = new_neuron

        return new_neuron

    def initialize_neurons_at_time(self, t=0, num_expected_neurons=None, df_global_tracks=None, verbose=0):
        """
        Each segmented neuron is initialized, even if there is not a tracklet at that particular volume

        Name corresponds to the list index of the raw neuron (at that time point),
        Note: this is offset by at least one from the segmentation ID label
        """
        # Instead of getting the neurons from the segmentation directly, get them from the global track dataframe
        self.logger.warning(f"Initializing at t={t}; if this is not the same as the template for the global track "
                            f"dataframe, then this initial tracklet may be incorrect")
        neurons_at_template = df_global_tracks.loc[[t], (slice(None), 'raw_neuron_ind_in_list')]
        neurons_in_global_df = get_names_from_df(neurons_at_template)
        # neuron_zxy = self.detections.get_neurons_at_time(t)
        # num_neurons = neuron_zxy.shape[0]
        num_neurons = len(neurons_in_global_df)
        print(f"Found {num_neurons} neurons")
        if num_expected_neurons and num_expected_neurons != num_neurons:
            self.logger.warning(f"Actual number of neurons ({num_neurons}) is not equal to the expected number "
                                f"at the template t={t} ({num_expected_neurons})")
            raise DataSynchronizationError("global track dataframe", "segmentation", "3a")

        new_tracklets = []
        for name_in_df in tqdm(neurons_in_global_df, total=num_neurons):
            new_neuron = self.initialize_new_neuron(initialization_frame=t, name=name_in_df)
            i_neuron_ind = int(df_global_tracks.loc[[t], (name_in_df, 'raw_neuron_ind_in_list')])
            # Add a tracklet if exists, otherwise create a length-1 tracklet to keep everything consistent
            _, tracklet_name = self.detections.get_tracklet_from_neuron_and_time(i_neuron_ind, t)

            if verbose >= 2:
                print(f"Initializing neuron named {name_in_df} and indexed {i_neuron_ind} at t={t}, "
                      f"with tracklet {tracklet_name} (None means a new tracklet is made)")
            if not tracklet_name:
                # Make a new tracklet, and give it data
                tracklet_name = self.detections.initialize_new_empty_tracklet()
                mask_ind = self.detections.segmentation_metadata.i_in_array_to_mask_index(t, i_neuron_ind)
                self.detections.update_tracklet_metadata_using_segmentation_metadata(
                    t=t, tracklet_name=tracklet_name, mask_ind=mask_ind, likelihood=1.0, verbose=self.verbose-1)
                new_tracklets.append(tracklet_name)

            tracklet = self.detections.df_tracklets_zxy[[tracklet_name]]
            confidence = 1.0
            new_neuron.add_tracklet(confidence, tracklet, metadata=tracklet_name)

        self.logger.info(f"Added new tracklets: {new_tracklets}")
        self.detections.synchronize_dataframe_to_disk()

    def initialize_neurons_using_previous_matches(self, previous_matches: Dict[str, List[str]]):
        """
        Matches should have the form:
        {'neuron_name': ['tracklet_name1', 'tracklet_name2', ...]}

        Parameters
        ----------
        previous_matches

        Returns
        -------

        """
        for name, matches in tqdm(previous_matches.items(), leave=False):
            new_neuron = self.initialize_new_neuron(name=name)

            confidence = 1.0
            for tracklet_name in matches:
                tracklet = self.detections.df_tracklets_zxy[[tracklet_name]]
                new_neuron.add_tracklet(confidence, tracklet, metadata=tracklet_name)

        if self.logger is not None:
            self.logger.info("Initialized using previous matches:")
            v = self.verbose
            self.verbose = 1
            self.logger.info(self)
            self.verbose = v

    def initialize_neurons_from_training_data(self, df_training_data):
        training_tracklet_names = translate_training_names_to_raw_names(df_training_data)
        # TODO: do these match up with the fdnc tracking names?
        # neuron_names = [int2name_neuron(i+1) for i in range(len(training_tracklet_names))]
        for i, name in enumerate(training_tracklet_names):
            # this tracklet should still have a multi-level index
            tracklet = self.detections.df_tracklets_zxy[[name]]
            initialization_frame = tracklet.first_valid_index()
            confidence = 1.0
            new_neuron = self.initialize_new_neuron(initialization_frame=initialization_frame)
            new_neuron.add_tracklet(confidence, tracklet, metadata=name)

    def add_previous_matches(self, previous_matches):
        df_tracklets = self.detections.df_tracklets_zxy
        verbose = self.verbose
        if previous_matches is not None:
            if self.logger is not None:
                self.logger.info(f"Found {len(previous_matches)} previously matched neurons")
            for neuron_name, match_names in previous_matches.items():
                neuron = self.global_name_to_neuron[neuron_name]
                for name in match_names:
                    previously_matched_tracklet = df_tracklets[[name]]
                    conf = 1.0  # Assume it was good
                    neuron.add_tracklet(conf, previously_matched_tracklet, metadata=name,
                                        check_using_classifier=False, verbose=verbose - 2)
        elif self.logger is not None:
            self.logger.info("No previous matches found")

    def initialize_all_neuron_tracklet_classifiers(self):
        for name, neuron in tqdm(self.global_name_to_neuron.items(), leave=False):
            list_of_tracklets = self.get_tracklets_for_neuron(name)
            neuron.initialize_tracklet_classifier(list_of_tracklets)

    def reinitialize_all_neurons_from_final_matching(self, final_matching: MatchesWithConfidence,
                                                     ignore_missing_tracklets=False):
        """Note: if there are originally neurons with no tracklet matches, then they should remain as they are"""
        self.backup_global_name_to_neuron()
        self.logger.debug(f"Before reinitialization: {self}")
        neuron2tracklet = final_matching.get_mapping_0_to_1()
        match2conf = final_matching.get_mapping_pair_to_conf()
        for neuron_name, tracklet_list in tqdm(neuron2tracklet.items()):
            new_neuron = NeuronComposedOfTracklets(neuron_name, initialization_frame=0, verbose=self.verbose - 1)
            for tracklet_name in tracklet_list:
                conf = match2conf[(neuron_name, tracklet_name)]
                try:
                    tracklet = self.detections.df_tracklets_zxy[[tracklet_name]]
                except KeyError as e:
                    if ignore_missing_tracklets:
                        self.logger.warning(f"Attempted match with {tracklet_name}, but it is not in the database")
                    else:
                        raise e
                new_neuron.add_tracklet(conf, tracklet, metadata=tracklet_name, check_using_classifier=False)
            self.global_name_to_neuron[neuron_name] = new_neuron
        self.logger.debug(f"After reinitialization: {self}")

    def backup_global_name_to_neuron(self):
        self.global_name_to_neuron_backup = copy.deepcopy(self.global_name_to_neuron)

    def tracks_with_gap_at_or_after_time(self, t) -> Dict[str, NeuronComposedOfTracklets]:
        return {name: neuron for name, neuron in self.global_name_to_neuron.items() if t > neuron.next_gap}

    def get_tracklets_and_network_names_for_neuron(self, neuron_name, minimum_confidence=None):
        list_of_tracklets = self.get_tracklets_for_neuron(neuron_name, minimum_confidence=minimum_confidence)
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_network_names = neuron.get_network_tracklet_names(minimum_confidence=minimum_confidence)
        return list_of_tracklets, tracklet_network_names

    def get_tracklets_for_neuron(self, neuron_name, minimum_confidence=None) -> List[pd.DataFrame]:
        """Proper null value is []"""
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_names = neuron.get_raw_tracklet_names(minimum_confidence=minimum_confidence)
        list_of_tracklets = [self.detections.df_tracklets_zxy[n] for n in tracklet_names]
        return list_of_tracklets

    def get_full_track_for_neuron(self, neuron_name) -> pd.DataFrame:
        list_of_tracklets = self.get_tracklets_for_neuron(neuron_name)
        if list_of_tracklets:
            df = list_of_tracklets[0].copy()
            for df2 in list_of_tracklets[1:]:
                df = df.combine_first(df2)
            return df
        else:
            return None

    def update_time_covering_ind_for_neuron(self, neuron_name):
        neuron = self.global_name_to_neuron[neuron_name]
        df = self.get_full_track_for_neuron(neuron_name)
        if df is not None:
            neuron.tracklet_covering_ind = list(df.dropna().index)
        else:
            neuron.tracklet_covering_ind = [neuron.initialization_frame]

    def update_time_covering_ind_for_all_neurons(self):
        for name in self.global_name_to_neuron.keys():
            self.update_time_covering_ind_for_neuron(name)

    def remove_conflicting_tracklets_from_all_neurons(self, verbose=0):
        for name in tqdm(self.global_name_to_neuron.keys()):
            self.remove_conflicting_tracklets_from_neuron(name, verbose=verbose-1)

    def remove_conflicting_tracklets_from_neuron(self, neuron_name, verbose=0):

        neuron = self.global_name_to_neuron[neuron_name]
        overlapping_confidences, overlapping_tracklet_names = \
            self.get_conflicting_tracklets_for_neuron(neuron_name)
        # Then just take the highest confidence one, removing all others
        # Note: this currently allows some multi-conflict tracklets to be in both "to_keep" and "to_remove"
        # As written, it just removes them anyway
        names_to_remove = set()
        names_to_keep = set()
        for names, confidences in zip(overlapping_tracklet_names, overlapping_confidences):
            i_sort = np.argsort(confidences)
            for i_to_remove in i_sort[:-1]:
                name_to_remove = names[i_to_remove]
                names_to_remove.add(name_to_remove)

        # edges_to_remove = list(zip(len(names_to_remove)*[neuron_network_name], names_to_remove))
        # neuron.neuron2tracklets.remove_edges_from(edges_to_remove)
        neuron.neuron2tracklets.remove_nodes_from(names_to_remove)

        if verbose >= 1:
            print(f"Removed {len(names_to_remove)} tracklets from {neuron.name}")
            print(f"Current neuron status: {neuron}")

    def get_conflicting_tracklets_for_neuron(self, neuron_name, minimum_confidence=None):
        tracklet_list, tracklet_network_names = self.get_tracklets_and_network_names_for_neuron(neuron_name,
                                                                                                minimum_confidence)
        neuron = self.global_name_to_neuron[neuron_name]
        # Loop through tracklets, and find conflicting sets
        overlapping_tracklet_names = get_names_of_conflicting_dataframes(tracklet_list, tracklet_network_names)
        overlapping_confidences = neuron.get_confidences_of_tracklets(overlapping_tracklet_names)
        return overlapping_confidences, overlapping_tracklet_names

    def get_conflict_times_for_tracklets_for_neuron(self, neuron_name, minimum_confidence=None, verbose=0):
        tracklet_list, tracklet_network_names = self.get_tracklets_and_network_names_for_neuron(neuron_name,
                                                                                                minimum_confidence)
        neuron = self.global_name_to_neuron[neuron_name]
        # Loop through tracklets, and find conflicting sets
        overlapping_tracklet_conflict_points = get_times_of_conflicting_dataframes(tracklet_list,
                                                                                   tracklet_network_names,
                                                                                   verbose=verbose)
        overlapping_tracklet_names = [list(overlapping_tracklet_conflict_points.keys())]
        overlapping_confidences = neuron.get_confidences_of_tracklets(overlapping_tracklet_names)
        return overlapping_confidences, overlapping_tracklet_conflict_points

    def get_conflict_time_dictionary_for_all_neurons(self, minimum_confidence=None):
        # TODO: check if some conflict times are too close
        overlapping_tracklet_conflict_points = {}
        for name in tqdm(self.neuron_names):
            _, these_conflicts = self.get_conflict_times_for_tracklets_for_neuron(name,
                                                                                  minimum_confidence=minimum_confidence)
            overlapping_tracklet_conflict_points.update(these_conflicts)
        return overlapping_tracklet_conflict_points

    def plot_tracklets_for_neuron(self, neuron_name, with_names=True, with_confidence=True, plot_field='z',
                                  diff_percentage=False, minimum_confidence=0.0, adjust_annotations=False):
        tracklet_list, tracklet_network_names = self.get_tracklets_and_network_names_for_neuron(neuron_name,
                                                                                                minimum_confidence)
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_names = neuron.get_raw_tracklet_names(minimum_confidence=minimum_confidence)
        num_lines = len(tracklet_names)

        fig = plt.figure(figsize=(45, 5))
        num_skipped = 0
        all_annotations = []
        for i, (t, name) in enumerate(tqdm(zip(tracklet_list, tracklet_names))):
            edge = (neuron.name_in_graph, neuron.neuron2tracklets.raw_name_to_network_name(name))
            conf = neuron.neuron2tracklets.get_edge_data(*edge)['weight']

            if conf < minimum_confidence:
                num_skipped += 1
                continue

            y = t[plot_field]
            if diff_percentage:
                y = y.diff() / y
            line = plt.plot(y)

            if with_names or with_confidence:
                x0 = y.first_valid_index()
                jitter = 0.2*np.max(y)*np.random.rand() - 0.1*np.max(y)
                if (i - num_skipped) % 2 == 0:
                    y_text = (i / num_lines) * np.min(y) + jitter
                else:
                    y_text = (i / num_lines) * np.min(y) + np.max(y) + jitter
                y0 = y.at[x0]
                if with_names:
                    annotation_str = name
                else:
                    annotation_str = ""
                if with_confidence:
                    annotation_str = f"{annotation_str} conf={conf:.2f}"
                txt = plt.annotate(annotation_str, (x0, y0), xytext=(x0-1, y_text),
                                   arrowprops=dict(facecolor=line[0].get_color()))
                ylim = plt.gca().get_ylim()
                plt.ylim([0, 2*np.max(y)])
                all_annotations.append(txt)
        plt.title(f"Tracklets for {neuron_name}")

        if adjust_annotations and len(all_annotations) > 0:
            from adjustText import adjust_text
            print("Adjusting annotations to not overlap...")
            adjust_text(all_annotations, only_move={'points': 'y', 'text': 'y', 'objects': 'y'})

        plt.ylabel(plot_field)

        return fig

    def compose_global_neuron_and_tracklet_graph(self) -> MatchesAsGraph:
        return nx.compose_all([neuron.neuron2tracklets for neuron in self.global_name_to_neuron.values()])

    @property
    def num_total_matched_tracklets(self):
        total = 0
        for name in self.neuron_names:
            total += self.global_name_to_neuron[name].num_matched_tracklets
        return total

    def __repr__(self):
        short_message = f"Worm with {self.num_neurons} neurons"
        if self.verbose == 0:
            return short_message
        else:
            return f"{short_message} and " \
                   f"{self.num_total_matched_tracklets}/{self.detections.num_total_tracklets} matched tracklets"
