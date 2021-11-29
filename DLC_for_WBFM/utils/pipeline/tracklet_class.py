from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.pipeline.matches_class import MatchesWithConfidence
from DLC_for_WBFM.utils.projects.utils_filepaths import lexigraphically_sort
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors


@dataclass
class TrackedNeuron:

    name: str
    initialization_frame: int

    tracklet_matches: MatchesWithConfidence = MatchesWithConfidence()
    tracklet_covering_ind: list = None

    verbose: int = 0

    def __post_init__(self):
        if self.tracklet_covering_ind is None:
            self.tracklet_covering_ind = []

    # For use when assigning matches and iterating over time
    @property
    def next_gap(self):
        return self.tracklet_covering_ind[-1] + 1

    def add_tracklet(self, tracklet_name, confidence, tracklet: pd.DataFrame):
        self.tracklet_matches.add_match([self.name, tracklet_name, confidence])

        tracklet_covering = np.where(tracklet['z'].notnull())[0]
        self.tracklet_covering_ind.extend(tracklet_covering)
        # self.next_gap = tracklet_covering[-1] + 1

        if self.verbose >= 2:
            print(f"Added tracklet {tracklet_name} to neuron {self.name} with next gap currently: {self.next_gap}")


@dataclass
class TrackedWorm:

    neurons: Dict[str, TrackedNeuron] = None

    verbose: int = 0

    @property
    def num_neurons(self):
        return len(self.neurons)

    def get_next_neuron_name(self):
        return int2name(self.num_neurons + 1)

    def __post_init__(self):
        if self.neurons is None:
            self.neurons = {}

    def initialize_new_neuron(self, name=None, initialization_frame=0):
        if name is None:
            name = self.get_next_neuron_name()

        new_neuron = TrackedNeuron(name, initialization_frame, verbose=self.verbose-1)
        self.neurons[name] = new_neuron

        return new_neuron

    def tracks_with_gap_at_or_after_time(self, t):
        return {name: neuron for name, neuron in self.neurons.items() if t > neuron.next_gap}


@dataclass
class TrackletDictionary:

    data: pd.DataFrame
    segmentation_metadata: DetectedNeurons

    def get_closest_tracklet_to_point(self,
                                      i_time,
                                      target_pt,
                                      nbr_obj: NearestNeighbors = None,
                                      nonnan_ind=None,
                                      verbose=0):
        df_tracklets = self.data
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
