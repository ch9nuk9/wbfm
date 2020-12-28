import h5py

##
## Reading DeepLabCut
##

def xy_from_dlc_dat(fname, which_neuron=0, num_frames=100):

    xy_ind = range(which_neuron*3, which_neuron*3 + 2)
    prob_ind = which_neuron*3 + 2

    this_xy = []
    this_prob = []

    with h5py.File(fname, 'r') as dlc_dat:
        dlc_table = dlc_dat['df_with_missing']['table']
        which_frames = range(num_frames)
        this_xy.extend([this_frame[1][xy_ind] for this_frame in dlc_table[which_frames]])
        this_prob.extend([this_frame[1][prob_ind] for this_frame in dlc_table[which_frames]])

    return this_xy, this_prob


def get_number_of_annotations(annotation_fname):
    # Get the number of neurons and frames annotated
    with h5py.File(annotation_fname, 'r') as dlc_dat:
        dlc_table = dlc_dat['df_with_missing']['table']
        # Each table entry has: x, y, probability
        num_neurons = len(dlc_table[0][1])//3
        which_neurons = range(num_neurons)
        num_frames = len(dlc_table)
    print(f'Found annotations for {num_neurons} neurons and {num_frames} frames')

    return num_neurons, which_neurons, num_frames
