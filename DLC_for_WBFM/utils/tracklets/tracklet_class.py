import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import dataframe_to_dataframe_zxy_format, get_names_from_df, \
    get_names_of_conflicting_dataframes
from DLC_for_WBFM.utils.neuron_matching.matches_class import MatchesAsGraph, MatchesWithConfidence
from DLC_for_WBFM.utils.projects.utils_filenames import lexigraphically_sort
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors

from DLC_for_WBFM.utils.tracklets.tracklet_to_DLC import translate_training_names_to_raw_names


@dataclass
class NeuronComposedOfTracklets:

    name: str = None
    initialization_frame: int = None
    # initialization_neuron_name: str

    neuron2tracklets: MatchesAsGraph = None
    tracklet_covering_ind: list = None

    # For detecting outliers in candidate additional tracklet matches
    classifier: sklearn.svm._classes.OneClassSVM = None
    scaler: sklearn.preprocessing._data.StandardScaler = None
    classifier_rejection_threshold: float = 0.0
    fields_to_classify: list = None

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
            if verbose >= 1:
                print(f"{tracklet_name} did not pass classifier check")
            return False

        is_match_added = self.neuron2tracklets.add_match_if_not_present([self.neuron_ind, i_tracklet, confidence],
                                                                        node0_metadata=self.name,
                                                                        node1_metadata=tracklet_name,
                                                                        edge_metadata=metadata)
        if is_match_added:
            tracklet_covering = np.where(tracklet[tracklet_name]['z'].notnull())[0]
            self.tracklet_covering_ind.extend(tracklet_covering)
            # self.next_gap = tracklet_covering[-1] + 1

            if self.verbose >= 2:
                print(f"Added tracklet {i_tracklet} to neuron {self.name} with next gap: {self.next_gap}")

        return is_match_added

    def initialize_tracklet_classifier(self, list_of_tracklets, min_pts=10):
        """This object doesn't see the raw tracklet data, so it must be sent in the call"""

        x = [tracklet[self.fields_to_classify].dropna().to_numpy() for tracklet in list_of_tracklets]
        if len(x) > 1:
            x = np.vstack(x)
        else:
            x = x[0]
        assert x.shape[0] >= min_pts, "Neuron needs more points to build a classifier"

        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        self.classifier = OneClassSVM(nu=0.05, gamma=0.05, kernel='rbf').fit(x)

    def check_new_tracklet_using_classifier(self, candidate_tracklet: pd.DataFrame):
        y = candidate_tracklet[self.fields_to_classify]
        y = self.scaler.transform(y.values)
        predictions = self.classifier.predict(y)  # -1 means outlier
        fraction_outliers = np.mean(predictions)
        if fraction_outliers < self.classifier_rejection_threshold:
            return False, fraction_outliers
        else:
            return True, fraction_outliers

    def get_raw_tracklet_names(self):
        network_names = self.get_network_tracklet_names()
        nodes = self.neuron2tracklets.nodes()
        tracklet_names = [nodes[n]['metadata'] for n in network_names]
        return tracklet_names

    def get_network_tracklet_names(self):
        network_names = self.neuron2tracklets.get_all_matches(name=self.name_in_graph)
        return network_names

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

    def __repr__(self):
        return f"Neuron {self.name} (index={self.neuron_ind}) with {len(self.neuron2tracklets) - 1} tracklets " \
               f"from time {self.initialization_frame} to {self.next_gap}"


@dataclass
class DetectedTrackletsAndNeurons:

    df_tracklets_zxy: pd.DataFrame
    segmentation_metadata: DetectedNeurons

    local_neuron_to_tracklet: MatchesAsGraph = None
    df_tracklet_matches: pd.DataFrame = None  # Custom dataframe format containing raw neuron indices

    def __post_init__(self):
        if self.df_tracklet_matches is not None:
            local_neuron_to_tracklet = MatchesAsGraph(offset_convention=[True, False],
                                                      naming_convention=['neuron', 'tracklet'],
                                                      name_prefixes=['frame', 'trackletGroup'])

            for i, row in tqdm(self.df_tracklet_matches.iterrows(), total=len(self.df_tracklet_matches)):
                i_tracklet = int(row['clust_ind'])
                for i_local_frame, (i_local_neuron, i_global_frame) in enumerate(
                        zip(row['all_ind_local'], row['slice_ind'])):
                    try:
                        conf = row['all_prob'][i_local_frame]
                    except IndexError:
                        conf = np.nan
                    local_neuron_to_tracklet.add_match_if_not_present([i_local_neuron, i_tracklet, conf],
                                                                      group_ind0=i_global_frame)

    @property
    def all_tracklet_names(self):
        return self.df_tracklets_zxy.columns.get_level_values(0).drop_duplicates()

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

        # TODO: Directly use the neuron id - tracklet id matching dataframe
        target_pt = self.segmentation_metadata.mask_index_to_zxy(i_time, seg_ind)
        return self.get_closest_tracklet_to_point(i_time, target_pt)

    def get_neuron_index_within_tracklet(self, i_tracklet, t_local):
        this_tracklet = self.df_tracklet_matches.loc[i_tracklet]
        return this_tracklet['all_ind_local'][t_local]

    def get_tracklet_from_neuron_and_time(self, i_local_neuron, i_time):
        df = self.df_tracklets_zxy
        mask = df.loc[i_time, (slice(None), 'raw_neuron_id')] == i_local_neuron
        try:
            ind = np.where(mask)[0][0]
            return ind, self.all_tracklet_names[ind]
        except IndexError:
            return None, None

    def __repr__(self):
        return f"DetectedTrackletsAndNeurons object with {len(self.all_tracklet_names)} tracklets"


@dataclass
class TrackedWorm:

    global_name_to_neuron: Dict[str, NeuronComposedOfTracklets] = None
    global_name_to_neuron_backup: Dict[str, NeuronComposedOfTracklets] = None
    detections: DetectedTrackletsAndNeurons = None

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

    def initialize_neurons_at_time(self, t=0):
        for i, name in enumerate(self.detections.all_tracklet_names):
            # this tracklet should still have a multi-level index
            tracklet = self.detections.df_tracklets_zxy[[name]]
            # Assume tracklets are ordered, such that the first tracklet which starts at t>0 mean all the rest do
            if np.isnan(tracklet[name]['z'].iloc[t]):
                break
            confidence = 1.0
            new_neuron = self.initialize_new_neuron(initialization_frame=t)
            new_neuron.add_tracklet(confidence, tracklet, metadata=name)

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

    def initialize_all_neuron_tracklet_classifiers(self):
        for name, neuron in tqdm(self.global_name_to_neuron.items(), leave=False):
            list_of_tracklets = self.get_tracklets_for_neuron(name)
            neuron.initialize_tracklet_classifier(list_of_tracklets)

    def reinitialize_all_neurons_from_final_matching(self, final_matching: MatchesWithConfidence):
        self.backup_global_name_to_neuron()
        # TODO: don't just overwrite old neurons
        neuron2tracklet = final_matching.get_mapping_0_to_1()
        match2conf = final_matching.get_mapping_pair_to_conf()
        for neuron_name, tracklet_list in tqdm(neuron2tracklet.items()):
            new_neuron = NeuronComposedOfTracklets(neuron_name, initialization_frame=0, verbose=self.verbose - 1)
            for tracklet_name in tracklet_list:
                conf = match2conf[(neuron_name, tracklet_name)]
                tracklet = self.detections.df_tracklets_zxy[[tracklet_name]]
                new_neuron.add_tracklet(conf, tracklet, metadata=tracklet_name, check_using_classifier=False)
            self.global_name_to_neuron[neuron_name] = new_neuron

    def backup_global_name_to_neuron(self):
        self.global_name_to_neuron_backup = copy.deepcopy(self.global_name_to_neuron)

    def tracks_with_gap_at_or_after_time(self, t) -> Dict[str, NeuronComposedOfTracklets]:
        return {name: neuron for name, neuron in self.global_name_to_neuron.items() if t > neuron.next_gap}

    def get_tracklets_for_neuron(self, neuron_name) -> List[pd.DataFrame]:
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_names = neuron.get_raw_tracklet_names()
        list_of_tracklets = [self.detections.df_tracklets_zxy[n] for n in tracklet_names]
        return list_of_tracklets

    def get_full_track_for_neuron(self, neuron_name) -> pd.DataFrame:
        list_of_tracklets = self.get_tracklets_for_neuron(neuron_name)
        df = list_of_tracklets[0].copy()
        for df2 in list_of_tracklets[1:]:
            df = df.combine_first(df2)
        return df

    def update_time_covering_ind_for_neuron(self, neuron_name):
        neuron = self.global_name_to_neuron[neuron_name]
        df = self.get_full_track_for_neuron(neuron_name)
        neuron.tracklet_covering_ind = list(df.dropna().index)

    def update_time_covering_ind_for_all_neurons(self):
        for name in self.global_name_to_neuron.keys():
            self.update_time_covering_ind_for_neuron(name)

    def remove_conflicting_tracklets_from_all_neurons(self, verbose=0):
        for name in self.global_name_to_neuron.keys():
            self.remove_conflicting_tracklets_from_neuron(name, verbose=verbose-1)

    def remove_conflicting_tracklets_from_neuron(self, neuron_name, verbose=0):

        neuron = self.global_name_to_neuron[neuron_name]
        overlapping_confidences, overlapping_tracklet_names = \
            self.get_conflicting_tracklets_for_neuron(neuron_name)
        # Then just take the highest confidence one, removing all others
        # TODO: this currently allows some multi-conflict tracklets to be in both "to_keep" and "to_remove"
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

    def get_conflicting_tracklets_for_neuron(self, neuron_name):
        tracklet_list = self.get_tracklets_for_neuron(neuron_name)
        neuron = self.global_name_to_neuron[neuron_name]
        neuron_network_name = neuron.name_in_graph
        tracklet_network_names = neuron.get_network_tracklet_names()
        # Loop through tracklets, and find conflicting sets
        overlapping_tracklet_names = get_names_of_conflicting_dataframes(tracklet_list, tracklet_network_names)
        overlapping_confidences = []
        for these_names in overlapping_tracklet_names:
            overlapping_confidences.append(
                [neuron.neuron2tracklets.get_edge_data(neuron_network_name, t)['weight'] for t in these_names])
        return overlapping_confidences, overlapping_tracklet_names

    def plot_tracklets_for_neuron(self, neuron_name, with_names=True, with_confidence=True, plot_field='z',
                                  diff_percentage=False):
        tracklet_list = self.get_tracklets_for_neuron(neuron_name)
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_names = neuron.get_raw_tracklet_names()

        plt.figure(figsize=(25, 5))
        for t, name in zip(tracklet_list, tracklet_names):
            y = t[plot_field]
            if diff_percentage:
                y = y.diff() / y
                # err
            plt.plot(y)
            plt.ylabel(plot_field)

            if with_names or with_confidence:
                x0 = y.first_valid_index()
                y0 = y.at[x0]
                if with_names:
                    annotation_str = name
                else:
                    annotation_str = ""
                if with_confidence:
                    edge = (neuron.name_in_graph, neuron.neuron2tracklets.raw_name_to_network_name(name))
                    conf = neuron.neuron2tracklets.get_edge_data(*edge)['weight']
                    annotation_str = f"{annotation_str} conf={conf}"
                plt.annotate(annotation_str, (x0, y0))
        plt.title(f"Tracklets for {neuron_name}")

    def compose_global_neuron_and_tracklet_graph(self) -> MatchesAsGraph:
        return nx.compose_all([neuron.neuron2tracklets for neuron in self.global_name_to_neuron.values()])

    def __repr__(self):
        short_message = f"Worm with {self.num_neurons} neurons"
        if self.verbose == 0:
            return short_message
        else:
            return f"{short_message}"
