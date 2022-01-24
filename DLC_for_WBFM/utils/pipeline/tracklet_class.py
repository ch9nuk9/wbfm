import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import dataframe_to_dataframe_zxy_format
from DLC_for_WBFM.utils.pipeline.matches_class import MatchesAsGraph
from DLC_for_WBFM.utils.projects.utils_filenames import lexigraphically_sort
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors

from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import translate_training_names_to_raw_names


@dataclass
class NeuronComposedOfTracklets:

    name: str = None
    initialization_frame: int = None
    # initialization_neuron_name: str

    neuron2tracklets: MatchesAsGraph = None
    tracklet_covering_ind: list = None

    # For detecting outliers in candidate additional tracklet matches
    base_classifier: sklearn.svm._classes.OneClassSVM = None
    classifier_rejection_threshold: float = 0.0
    fields_to_classify: list = None

    verbose: int = 0

    @property
    def neuron_ind(self):
        return name2int_neuron_and_tracklet(self.name) - 1

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

    def add_tracklet(self, i_tracklet, confidence, tracklet: pd.DataFrame, metadata=None,
                     check_using_classifier=False):
        tracklet_name = tracklet.columns.get_level_values(0).drop_duplicates()[0]
        passed_classifier_check = True
        if check_using_classifier:
            if self.base_classifier:
                passed_classifier_check = self.check_new_tracklet_using_classifier(tracklet[tracklet_name].dropna())
            else:
                logging.warning("Classifier requested but not initialized")
        if not passed_classifier_check:
            # logging.debug("Tracklet did not pass classifier check")
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
        self.base_classifier = OneClassSVM(nu=0.1).fit(x)

    def check_new_tracklet_using_classifier(self, candidate_tracklet):
        y = candidate_tracklet[self.fields_to_classify]
        predictions = self.base_classifier.predict(y)  # -1 means outlier
        if np.mean(predictions) < self.classifier_rejection_threshold:
            return False
        else:
            return True

    def get_raw_tracklet_names(self):
        this_neuron_name = self.neuron2tracklets.raw_name_to_network_name(self.name)
        network_names = self.neuron2tracklets.get_all_matches(name=this_neuron_name)
        nodes = self.neuron2tracklets.nodes()
        tracklet_names = [nodes[n]['metadata'] for n in network_names]
        return tracklet_names

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
        all_tracklet_names = lexigraphically_sort(list(df_tracklets.columns.levels[0]))

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
            new_neuron = self.initialize_new_neuron(initialization_frame=t)
            new_neuron.add_tracklet(i, 1.0, tracklet, metadata=f"Initial tracklet")

    def initialize_neurons_from_training_data(self, df_training_data):
        training_tracklet_names = translate_training_names_to_raw_names(df_training_data)
        # TODO: do these match up with the fdnc tracking names?
        # neuron_names = [int2name_neuron(i+1) for i in range(len(training_tracklet_names))]
        for i, name in enumerate(training_tracklet_names):
            # this tracklet should still have a multi-level index
            tracklet = self.detections.df_tracklets_zxy[[name]]
            initialization_frame = tracklet.first_valid_index()
            new_neuron = self.initialize_new_neuron(initialization_frame=initialization_frame)
            new_neuron.add_tracklet(i, 1.0, tracklet, metadata=f"Initial tracklet")

    def initialize_all_neuron_tracklet_classifiers(self):
        for name, neuron in self.global_name_to_neuron.items():
            list_of_tracklets = self.get_tracklets_for_neuron(name)
            neuron.initialize_tracklet_classifier(list_of_tracklets)

    def tracks_with_gap_at_or_after_time(self, t) -> Dict[str, NeuronComposedOfTracklets]:
        return {name: neuron for name, neuron in self.global_name_to_neuron.items() if t > neuron.next_gap}

    def get_tracklets_for_neuron(self, neuron_name) -> List[pd.DataFrame]:
        neuron = self.global_name_to_neuron[neuron_name]
        tracklet_names = neuron.get_raw_tracklet_names()
        list_of_tracklets = [self.detections.df_tracklets_zxy[n] for n in tracklet_names]
        return list_of_tracklets

    def compose_global_neuron_and_tracklet_graph(self) -> nx.Graph:
        return nx.compose_all([g.neuron2tracklets for g in self.global_name_to_neuron.values()])

    def __repr__(self):
        short_message = f"Worm with {self.num_neurons} neurons"
        if self.verbose == 0:
            return short_message
        else:
            return f"{short_message}"
