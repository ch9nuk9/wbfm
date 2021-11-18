from dataclasses import dataclass

import numpy as np
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchesWithConfidence:

    indices1: list
    indices2: list
    confidence: list = None

    @property
    def names1(self):
        return [int2name(i + 1) for i in self.indices1]

    @property
    def names2(self):
        return [int2name(i + 1) for i in self.indices2]

    @property
    def mapping_0_to_1(self):
        return {n0: n1 for n0, n1 in zip(self.indices1, self.indices2)}

    @property
    def mapping_1_to_0(self):
        return {n1: n0 for n0, n1 in zip(self.indices1, self.indices2)}

    @property
    def mapping_0_to_1_names(self):
        return {n0: n1 for n0, n1 in zip(self.names1, self.names2)}

    @property
    def mapping_1_to_0_names(self):
        return {n1: n0 for n0, n1 in zip(self.names1, self.names2)}

    @property
    def mapping_pair_to_conf(self):
        return {(n0, n1): c for n0, n1, c in zip(self.indices1, self.indices2, self.likelihoods)}

    @property
    def mapping_pair_to_conf_names(self):
        return {(n0, n1): c for n0, n1, c in zip(self.names1, self.names2, self.likelihoods)}

    @property
    def matches_with_conf(self):
        return np.array(np.hstack([self.indices1, self.indices2, self.likelihoods]))

    @staticmethod
    def matches_from_array(matches_with_conf):
        i1 = [int(m) for m in matches_with_conf[:, 0]]
        i2 = [int(m) for m in matches_with_conf[:, 1]]
        return MatchesWithConfidence(i1, i2, matches_with_conf[:, 2])

    @staticmethod
    def matches_from_distance_matrix(dist):
        row_i, col_i = linear_sum_assignment(dist)
        conf = [dist[i, j] for i, j in zip(row_i, col_i)]
        return MatchesWithConfidence(row_i, col_i, conf)
