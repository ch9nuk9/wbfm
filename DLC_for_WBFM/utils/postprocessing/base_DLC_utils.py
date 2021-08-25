import h5py
import numpy as np
import pandas as pd


##
## Reading DeepLabCut
##

def xy_from_dlc_dat(fname, which_neuron=0, num_frames=100):
    xy_ind = range(which_neuron * 3, which_neuron * 3 + 2)
    prob_ind = which_neuron * 3 + 2

    try:
        with h5py.File(fname, 'r') as dlc_dat:
            dlc_table = dlc_dat['df_with_missing']['table']
            which_frames = range(num_frames)
            this_xy = []
            this_prob = []
            this_xy.extend([this_frame[1][xy_ind] for this_frame in dlc_table[which_frames]])
            this_prob.extend([this_frame[1][prob_ind] for this_frame in dlc_table[which_frames]])
    except KeyError:
        # TODO: check if I need the above
        dlc_table = pd.read_hdf(fname)
        scorer = 'feature_tracker'  # WARNING: SHOULDN'T BE HARD CODED
        neuron_names = dlc_table[scorer].columns.levels[0]
        name = neuron_names[which_neuron]

        this_xy = np.array(dlc_table[scorer][name].iloc[:, :2])
        this_prob = np.array(dlc_table[scorer][name].iloc[:, -1])

    return this_xy, this_prob


def get_number_of_annotations(annotation_fname):
    # Get the number of neurons and frames annotated
    try:
        with h5py.File(annotation_fname, 'r') as dlc_dat:
            dlc_table = dlc_dat['df_with_missing']['table']
            # Each table entry has: x, y, probability
            num_neurons = len(dlc_table[0][1]) // 3
            which_neurons = range(num_neurons)
            num_frames = len(dlc_table)
    except KeyError:
        dlc_table = pd.read_hdf(annotation_fname)
        num_neurons = len(dlc_table.columns) // 3
        which_neurons = range(num_neurons)
        num_frames = len(dlc_table)
    # print(f'Found annotations for {num_neurons} neurons and {num_frames} frames')

    return num_neurons, which_neurons, num_frames
