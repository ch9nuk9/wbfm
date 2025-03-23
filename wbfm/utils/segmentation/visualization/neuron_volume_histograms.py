import wbfm.utils.segmentation.util.overlap as ol
import os
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt


def plot_neuron_volume_histogram(volumes, save_path='', save_flag=0):
    # plot some magic
    vals = volumes.values()
    keys = volumes.keys()

    fig = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.hist(vals, bins=50)
    plt.xticks(np.arange(1, max(vals), 100))
    plt.xlabel('neuron volumes')
    plt.ylabel('# of neurons')
    plt.title('neuron volumes histogram')
    if save_path:
        tit1 = os.path.split(save_path)[1] + ' Volumes histogram'
        plt.title(tit1)

    if save_flag >= 1:
        plt.savefig(os.path.join(save_path + '_neuron_volumes_histogram.png'), dpi=96)

    fig1 = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.bar(keys, vals)
    plt.title('Volumes per neuron')
    if save_path:
        tit2 = os.path.split(save_path)[1] + ' Volumes per neuron'
        plt.title(tit2)
    plt.ylabel('Volume (pixels)')
    plt.xlabel('Neuron #')

    if save_flag >= 1:
        plt.savefig(os.path.join(save_path + '_neuron_volumes_barplot.png'), dpi=96)

    return fig, fig1


# create list of results
algo_path = r'C:\Segmentation_working_area\stitched_3d_data'
sv_base = r'C:\Segmentation_working_area\neuron_volumes'
files = [os.path.join(algo_path, f.name) for f in os.scandir(algo_path) if f.is_file() and 'gt' in f.name]

for file in files:
    print(f'Current file: {file}')

    this_array = np.load(file)

    if this_array.shape[0] == 33:
        this_array = this_array[1:]

    neurons = np.unique(this_array)
    neurons = np.delete(neurons, np.where(neurons == 0))
    volume = defaultdict(int)

    fname = os.path.split(file)[-1][:-4]
    sv_path = os.path.join(sv_base, fname)
    sv_file = os.path.join(sv_path, fname + '_volumes.pickle')

    if not os.path.isdir(sv_path):
        os.mkdir(sv_path)

    for n in neurons:
        volume[n] = np.count_nonzero(this_array == n)

    with open(sv_file, 'wb') as save:
        pickle.dump(volume, save)

    h1, h2 = plot_neuron_volume_histogram(volume, os.path.join(sv_path, fname), 1)



