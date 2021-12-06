from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from networkx import Graph

from DLC_for_WBFM.utils.feature_detection.utils_networkx import dist2conf
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name, int2name_using_mode
from scipy.optimize import linear_sum_assignment


def reverse_dict(d):
    return {v: k for k, v in d.items()}


@dataclass
class MatchesWithConfidence:

    indices0: list = None
    indices1: list = None
    confidence: list = None

    dist2conf_gamma: float = None
    indices_have_offset: List[bool] = None
    int2name_funcs: List[callable] = None

    # TODO
    reason_for_matches: list = None

    @property
    def names0(self):
        if self.indices_have_offset[0]:
            return [self.int2name_funcs[0](i + 1) for i in self.indices0]
        else:
            return [self.int2name_funcs[0](i) for i in self.indices0]

    @property
    def names1(self):
        if self.indices_have_offset[1]:
            return [self.int2name_funcs[1](i + 1) for i in self.indices1]
        else:
            return [self.int2name_funcs[1](i) for i in self.indices1]

    def __post_init__(self):
        if self.indices0 is None:
            self.indices0 = []
        if self.indices1 is None:
            self.indices1 = []
        if self.confidence is None:
            self.confidence = []
        if self.int2name_funcs is None:
            self.int2name_funcs = [int2name, int2name]
        try:
            # Should be two element list, but may be passed as a bool
            if len(self.indices_have_offset) == 1:
                self.indices_have_offset = [self.indices_have_offset[0], self.indices_have_offset[0]]
        except TypeError:
            self.indices_have_offset = [self.indices_have_offset, self.indices_have_offset]

    def add_match(self, new_match):
        assert len(new_match) == 3

        self.indices0.append(new_match[0])
        self.indices1.append(new_match[1])
        self.confidence.append(new_match[2])

    def get_mapping_0_to_1(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n0: n1 for n0, n1 in zip(self.indices0, self.indices1)}
        else:
            return {n0: n1 for n0, n1, c in zip(self.indices0, self.indices1, self.confidence) if c > conf_threshold}

    def get_mapping_1_to_0(self, conf_threshold=0.0):
        return reverse_dict(self.get_mapping_0_to_1(conf_threshold))

    def get_mapping_0_to_1_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n0: n1 for n0, n1 in zip(self.names0, self.names1)}
        else:
            return {n0: n1 for n0, n1, c in zip(self.names0, self.names1, self.confidence) if c > conf_threshold}

    def get_mapping_1_to_0_names(self, conf_threshold=0.0):
        return reverse_dict(self.get_mapping_0_to_1_names(conf_threshold))

    def get_mapping_pair_to_conf(self, conf_threshold=0.0):
        if self.confidence is None:
            return None
        else:
            return {(n0, n1): c for n0, n1, c in zip(self.indices0, self.indices1, self.confidence)
                    if c > conf_threshold}

    def get_mapping_0_to_1_with_unmatched_names(self, conf_threshold=0.0):
        # Generates a new name with suffix '_unmatched' if the neuron is found in index0, but has no match
        mapping = self.get_mapping_0_to_1_names(conf_threshold)
        update_mapping = {n0: f"{n0}_unmatched" for n0 in self.names0 if n0 not in mapping}
        mapping.update(update_mapping)
        return mapping

    def get_mapping_1_to_0_with_unmatched_names(self, conf_threshold=0.0):
        mapping = self.get_mapping_1_to_0_names(conf_threshold)
        update_mapping = {n1: f"{n1}_unmatched" for n1 in self.names1 if n1 not in mapping}
        mapping.update(update_mapping)
        return mapping

    def get_mapping_pair_to_conf_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return None
        else:
            return {(n0, n1): c for n0, n1, c in zip(self.names0, self.names1, self.confidence)
                    if c > conf_threshold}

    def get_num_matches(self, conf_threshold=0.0):
        return len(self.get_mapping_0_to_1(conf_threshold))

    @property
    def matches_with_conf(self):
        return np.array(np.hstack([self.indices0, self.indices1, self.confidence]))

    @staticmethod
    def matches_from_array(matches_with_conf):
        i1 = [int(m) for m in matches_with_conf[:, 0]]
        i2 = [int(m) for m in matches_with_conf[:, 1]]
        return MatchesWithConfidence(i1, i2, matches_with_conf[:, 2])

    @staticmethod
    def matches_from_distance_matrix(dist, gamma=1.0):
        row_i, col_i = linear_sum_assignment(dist)
        all_dist = [dist[i, j] for i, j in zip(row_i, col_i)]
        conf = dist2conf(np.array(all_dist), gamma=gamma)
        return MatchesWithConfidence(row_i, col_i, conf, gamma)

    def __repr__(self):
        return f"MatchesWithConfidence class with {self.get_num_matches()} matches"


class MatchesAsGraph(Graph):
    ind2names: Dict[Tuple, str]  # Tuple notation is (frame #, neuron #)

    offset_convention: List[bool] = None  # Whether has offset or not
    naming_convention: List[str] = None  # Will name nodes to keep them unique; can also be tracklet
    name_prefixes: List[str] = None

    def __init__(self, ind2names=None, name_prefixes=None, naming_convention=None, offset_convention=None):
        if ind2names is None:
            ind2names = {}
        if name_prefixes is None:
            self.name_prefixes = ['frame', 'frame']
        if naming_convention is None:
            self.naming_convention = ['neuron', 'neuron']
        if offset_convention is None:
            self.offset_convention = [True, True]

        self.ind2names = ind2names

        super().__init__()

    @property
    def names2ind(self):
        return {v: k for k, v in self.ind2names.items()}

    def __contains__(self, n):
        if self.has_node(n):  # Calls the super class
            return True
        else:
            try:
                if self.has_node(self.tuple2name(n[0], n[1])):
                    return True
            except IndexError:
                return False
            finally:
                return False

    def tuple2name(self, group_ind, ind):
        if self.offset_convention[group_ind]:
            ind += 1
        name = int2name_using_mode(ind, self.naming_convention[group_ind])
        return f"{self.name_prefixes[group_ind]}_{group_ind}_{name}"

    def name2tuple(self, name):
        group_ind, ind = int(name.split('_')[1]), int(name.split('_')[3])
        if self.offset_convention[group_ind]:
            ind -= 1
        return group_ind, ind

    def add_match(self, new_match):
        assert len(new_match) == 3

        n0, n1, conf = new_match

        name0 = self.tuple2name(0, n0)
        name1 = self.tuple2name(1, n1)

        self.add_weighted_edges_from([(name0, name1, conf)])

    def get_match(self, group_and_ind=None, name=None):
        if name is not None and group_and_ind is not None:
            print("Specify either the indices as a tuple, or the full name, not both")
            raise NotImplementedError
        if group_and_ind is not None:
            name = self.tuple2name(*group_and_ind)

        matches = list(self.neighbors(name))
        if len(matches) > 1:
            print("More than one match found")
            raise NotImplementedError
        return matches[0]


