from dataclasses import dataclass

import numpy as np
from DLC_for_WBFM.utils.feature_detection.utils_networkx import dist2conf
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name
from scipy.optimize import linear_sum_assignment


def reverse_dict(d):
    return {v: k for k, v in d.items()}


@dataclass
class MatchesWithConfidence:

    indices0: list = None
    indices1: list = None
    confidence: list = None

    dist2conf_gamma: float = None

    indices_have_offset: bool = True

    # TODO
    reason_for_matches: list = None

    @property
    def names0(self):
        if self.indices_have_offset:
            return [int2name(i + 1) for i in self.indices0]
        else:
            return [int2name(i) for i in self.indices0]

    @property
    def names1(self):
        if self.indices_have_offset:
            return [int2name(i + 1) for i in self.indices1]
        else:
            return [int2name(i) for i in self.indices1]

    def __post_init__(self):
        if self.indices0 is None:
            self.indices0 = []
        if self.indices1 is None:
            self.indices1 = []
        if self.confidence is None:
            self.confidence = []

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
