from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.pipeline.matches_class import MatchesWithConfidence, MatchesAsGraph
from DLC_for_WBFM.utils.projects.utils_filepaths import lexigraphically_sort
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron, name2int_neuron, int2name_deprecated
from segmentation.util.utils_metadata import DetectedNeurons
from sklearn.neighbors import NearestNeighbors


@dataclass
class NeuronComposedOfTracklets:

    name: str
    initialization_frame: int
    # initialization_neuron_name: str

    neuron2tracklets: MatchesAsGraph = None
    tracklet_covering_ind: list = None

    verbose: int = 0

    @property
    def neuron_ind(self):
        return name2int_neuron(self.name) - 1

    def __post_init__(self):
        if self.tracklet_covering_ind is None:
            self.tracklet_covering_ind = []
        if self.neuron2tracklets is None:
            self.neuron2tracklets = MatchesAsGraph(offset_convention=[True, False],
                                                   naming_convention=['neuron', 'tracklet'],
                                                   name_prefixes=['frame', 'trackletGroup'])

    # For use when assigning matches and iterating over time
    @property
    def next_gap(self):
        return self.tracklet_covering_ind[-1] + 1

    def add_tracklet(self, i_tracklet, confidence, tracklet: pd.DataFrame, metadata=None):
        is_match_added = self.neuron2tracklets.add_match_if_not_present([self.neuron_ind, i_tracklet, confidence],
                                                                        metadata=metadata)

        if is_match_added:
            tracklet_covering = np.where(tracklet['z'].notnull())[0]
            self.tracklet_covering_ind.extend(tracklet_covering)
            # self.next_gap = tracklet_covering[-1] + 1

            if self.verbose >= 2:
                print(f"Added tracklet {i_tracklet} to neuron {self.name} with next gap: {self.next_gap}")

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
        df_tracklets = self.df_tracklets_zxy
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


@dataclass
class TrackedWorm:

    global_name_to_neuron: Dict[str, NeuronComposedOfTracklets] = None
    detections: DetectedTrackletsAndNeurons = None

    verbose: int = 0

    @property
    def num_neurons(self):
        return len(self.global_name_to_neuron)

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

    def tracks_with_gap_at_or_after_time(self, t) -> Dict[str, NeuronComposedOfTracklets]:
        return {name: neuron for name, neuron in self.global_name_to_neuron.items() if t > neuron.next_gap}

    def __repr__(self):
        short_message = f"Worm with {self.num_neurons} neurons"
        if self.verbose == 0:
            return short_message
        else:
            return f"{short_message}"

