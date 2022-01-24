import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from networkx import Graph, NetworkXError

from DLC_for_WBFM.utils.feature_detection.utils_networkx import dist2conf
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron, int2name_using_mode
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
            self.int2name_funcs = [int2name_neuron, int2name_neuron]
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
        if self.confidence is None:
            return {n1: n0 for n0, n1 in zip(self.indices0, self.indices1)}
        else:
            return {n1: n0 for n0, n1, c in zip(self.indices0, self.indices1, self.confidence) if c > conf_threshold}

    def get_mapping_0_to_1_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n0: n1 for n0, n1 in zip(self.names0, self.names1)}
        else:
            return {n0: n1 for n0, n1, c in zip(self.names0, self.names1, self.confidence) if c > conf_threshold}

    def get_mapping_1_to_0_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n1: n0 for n0, n1 in zip(self.names0, self.names1)}
        else:
            return {n1: n0 for n0, n1, c in zip(self.names0, self.names1, self.confidence) if c > conf_threshold}

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
        try:
            i1 = [int(m) for m in matches_with_conf[:, 0]]
            i2 = [int(m) for m in matches_with_conf[:, 1]]
        except TypeError:
            i1 = [int(m) for m in np.array(matches_with_conf[:, 0])]
            i2 = [int(m) for m in np.array(matches_with_conf[:, 1])]
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

    offset_convention: List[bool]  # Whether has offset or not
    naming_convention: List[str]  # Will name nodes to keep them unique; can also be tracklet
    name_prefixes: List[str]

    # node2raw_names: Dict[str, str] = None

    def __init__(self, ind2names=None, name_prefixes=None, naming_convention=None, offset_convention=None):
        if ind2names is None:
            ind2names = {}
        if name_prefixes is None:
            name_prefixes = ['frame', 'frame']
        if naming_convention is None:
            naming_convention = ['neuron', 'neuron']
        if offset_convention is None:
            offset_convention = [True, True]
        # if self.node2raw_names is None:
        #     self.node2raw_names = {}

        self.ind2names = ind2names
        self.name_prefixes = name_prefixes
        self.naming_convention = naming_convention
        self.offset_convention = offset_convention

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

    def tuple2name(self, bipartite_ind, local_ind, group_ind=None):
        """

        Parameters
        ----------
        bipartite_ind - Either 0 or 1; used with self.naming_convention and related fields
        ind - The second index; any integer, e.g. which neuron within a frame, or tracklet index
        group_ind - The first index, if different from bipartite_ind, e.g. which frame the neuron belongs to

        Returns
        -------

        Examples:
        if naming_convention == ['neuron', 'neuron']:
            bipartite_0_frame_1_neuron_001

        if subgroup_ind == 4:
            bipartite_1_frame_1_neuron_123

        """
        if self.offset_convention[bipartite_ind]:
            local_ind += 1
        naming_convention = self.naming_convention[bipartite_ind]
        if callable(naming_convention):
            name = naming_convention(local_ind)
        elif isinstance(naming_convention, str):
            name = int2name_using_mode(local_ind, naming_convention)
        else:
            raise TypeError(f"naming_convention must be callable or string; was {naming_convention}")
        prefix = f"bipartite_{bipartite_ind}_{self.name_prefixes[bipartite_ind]}"
        if group_ind is None:
            node_name = f"{prefix}_{bipartite_ind}_{name}"
        else:
            node_name = f"{prefix}_{bipartite_ind}_{name}"
        # NOTE: this name is actually generated new, thus may not be the true raw name
        # ... but should be, if the naming conventions are the same
        # self.node2raw_names[node_name] = name
        # INSTEAD: use the 'metadata' field of the nodes
        return node_name

    def name2tuple(self, name):
        # NOTE: doesn't tell you which bipartite element this came from
        # Sometimes this is the same as group_ind, but may not be
        n = self.nodes[name]
        bipartite_ind, group_ind, local_ind = n['bipartite'], n['group_ind'], n['local_ind']
        # bipartite_ind, group_ind, ind = int(name.split('_')[1]), int(name.split('_')[3]), int(name.split('_')[3])
        # if self.offset_convention[group_ind]:
        #     ind -= 1
        return bipartite_ind, group_ind, local_ind

    def add_match_if_not_present(self, new_match, group_ind0=0, group_ind1=1,
                                 node0_metadata=None,
                                 node1_metadata=None,
                                 edge_metadata=None):
        assert len(new_match) == 3

        n0, n1, conf = new_match

        name0 = self.tuple2name(bipartite_ind=0, group_ind=group_ind0, local_ind=n0)
        name1 = self.tuple2name(bipartite_ind=1, group_ind=group_ind1, local_ind=n1)

        if self.has_edge(name0, name1):
            return False
        else:
            self.add_node(name0, bipartite=0, group_ind=group_ind0, local_ind=n0, metadata=node0_metadata)
            self.add_node(name1, bipartite=1, group_ind=group_ind0, local_ind=n0, metadata=node1_metadata)
            self.add_weighted_edges_from([(name0, name1, conf)], metadata=edge_metadata)
            return True

    def get_unique_match(self, group_and_ind=None, name=None):
        name = self.process_query(group_and_ind, name)

        matches = self.get_all_matches(name=name)
        if matches is None:
            return None
        elif len(matches) > 1:
            print("More than one match found")
            raise NotImplementedError
        else:
            return matches[0]

    def get_all_matches(self, group_and_ind=None, name=None):
        name = self.process_query(group_and_ind, name)
        try:
            return list(self.neighbors(name))
        except NetworkXError:
            logging.debug("No match found")
            return None

    def process_query(self, group_and_ind, name):
        if name is not None and group_and_ind is not None:
            print("Specify either the indices as a tuple, or the full name, not both")
            raise NotImplementedError
        if group_and_ind is not None:
            name = self.tuple2name(*group_and_ind)
        return name

    def raw_name_to_network_name(self, raw_name):
        node_names = list(self)
        nodes = self.nodes(data=True)
        for name, node in zip(node_names, nodes):
            if raw_name == node['metadata']:
                return name
        else:
            return None

    def __repr__(self):
        return f"MatchesAsGraph object with {len(self.nodes)} nodes and {len(self.edges)} edges"


# def get_tracklet_name_from_full_name(name):
#     """Assume name is like: bipartite_1_trackletGroup_1_neuron228"""
#     return name.split('_')[-1]
