from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import numpy as np
import networkx as nx
import collections
from dataclasses import dataclass

##
## Basic class definition
##

@dataclass
class ReferenceFrame():
    """ Information for registered reference frames"""

    # Data for registration
    neuron_locs: list
    keypoints: list
    keypoint_locs: list # Just the z coordinate
    all_features: np.array
    features_to_neurons: dict

    # Metadata
    frame_ind: int = None
    video_fname: str = None
    vol_shape: tuple = None
    alpha: float = 1.0

    # To be finished with a set of other registered frames
    neuron_ids: list = None # global neuron index

    def iter_neurons(self):
        # Practice with yield
        for neuron in self.neuron_locs:
            yield neuron

    def get_features_of_neuron(self, which_neuron):
        iter_tmp = self.features_to_neurons.items()
        return [key for key,val in iter_tmp if val == which_neuron]
        #return np.argwhere(self.features_to_neurons == which_neuron)

    def num_neurons(self):
        return self.neuron_locs.shape[0]

    def get_data(self):
        return get_single_volume(self.video_fname,
                                 self.frame_ind,
                                 num_slices=self.vol_shape[0],
                                 alpha=self.alpha)


##
## Utilities for combining frames into a reference set
##

def get_node_name(frame_ind, neuron_ind):
    """The graph is indexed by integer, so all neurons must be unique"""
    return frame_ind*1000 + neuron_ind

def unpack_node_name(node_name):
    """Inverse of get_node_name"""
    return divmod(node_name, 1000)

def build_digraph_from_matches(pairwise_matches, pairwise_conf=None,
                              verbose=1):
    DG = nx.DiGraph()
    for frames, all_neurons in pairwise_matches.items():
        if verbose >= 1:
            print("==============================")
            print("Analyzing pair:")
            print(frames)
        if pairwise_conf is not None:
            all_conf = pairwise_conf[frames]
        else:
            all_conf = np.ones_like(np.array(all_neurons)[:,0])
        for neuron_pair, this_conf in zip(all_neurons, all_conf):
            #print(neuron_pair)
            node1 = get_node_name(frames[0], neuron_pair[0])
            node2 = get_node_name(frames[1], neuron_pair[1])
            e = (node1, node2, this_conf)
            DG.add_weighted_edges_from([e])

    return DG

##
## Related helper and visualization functions
##

def get_subgraph_with_strong_weights(DG, min_weight):
    #G = nx.from_numpy_matrix(DG, parallel_edges=False)
    G = DG.copy()
    edge_weights = nx.get_edge_attributes(G,'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w < min_weight))
    return G

def calc_connected_components(DG):
    all_neurons = list(nx.strongly_connected_components(DG))
    all_len = [len(c) for c in all_neurons]
    #print(all_len)
    big_comp = np.argmax(all_len)
    print("Largest connected component size: ", max(all_len))
    #print(big_comp)
    big_DG = DG.subgraph(all_neurons[big_comp])

    return big_DG, all_len

def plot_degree_hist(DG):
    import collections

    degree_sequence = sorted([d for n, d in DG.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")
