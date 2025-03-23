import wbfm.utils.segmentation.util.overlap as ol
import os
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

# create list of cp results
cp_path = r'C:\Segmentation_working_area\stitched_3d_data'
sv_path = r'C:\Segmentation_working_area\cellpose_testdata'
files = [os.path.join(cp_path, f.name) for f in os.scandir(cp_path) if f.is_file()]

for file in files:
    print(f'Current file: {file}')
    this_array = np.load(file)

    if this_array.shape[0] == 33:
        this_array = this_array[1:]

    neurons = np.unique(this_array)

    lengths = defaultdict(list)
    fname = os.path.split(file)[-1][:-4]
    sv_file = os.path.join(sv_path, fname, 'lengths.pickle')

    if not os.path.isdir(os.path.join(sv_path, fname)):
        os.mkdir(os.path.join(sv_path, fname))

    for neuron in neurons:
        if neuron == 0:
            continue
        z_count = 0

        for slice in range(len(this_array)):
            if neuron in this_array[slice]:
                z_count += 1

        lengths[neuron] = z_count

    with open(sv_file, 'wb') as save:
        pickle.dump(lengths, save)

    ol.neuron_length_hist(lengths, os.path.join(sv_path, fname), 1)
    plt.close('all')


def get_neuron_lengths_dict(arr):
    neurons = np.unique(this_array)
    lengths = defaultdict(list)

    for neuron in neurons:
        if neuron == 0:
            continue
        z_count = 0

        for slice in range(len(this_array)):
            if neuron in this_array[slice]:
                z_count += 1

        lengths[neuron] = z_count

    return lengths
