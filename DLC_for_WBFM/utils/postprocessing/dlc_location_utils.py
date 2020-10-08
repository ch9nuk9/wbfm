import h5py


def xy_from_dlc_dat(fname, which_neuron=0, num_frames=100):
    """
    Gets the xy locations of a single neuron from a DeepLabCut produced table
    """

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