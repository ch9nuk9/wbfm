"""
Plotting the results of IoU calculations.

Data/Input:
    - pickled list of IoUs: list of lists with 3 columns (gt neuron ID, algorithm neuron ID, IoU in %)
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# TODO: cycle over all subfolders

dirs = [d.name for d in os.scandir('../..') if d.is_dir()]      # list of subdirectories

# Parent folder. Maybe change all of thsi script into a function and then use arg1 as parent!
# for now: get current working directory
parent = os.getcwd()

# iterate over folders
for folder in dirs:
    cwd = os.join.path(parent, folder)
    os.chdir(cwd)

    # Get a filename for the data
    dat_folder = 'dat'
    fname = 'list_of_ious.pickle'
    # folder "dat" only exists in our examples (for now)
    if os.path.isdir(os.path.join(cwd, dat_folder)):
        fname = os.path.join(dat_folder, fname)
    else:
        fname = os.path.join(cwd, fname)

    # Import the data
    with open(fname,'rb') as file:
        dat = pickle.load(file)

    # Check the sizes and whatnot
    # 3 columns: neuron index of gt, index of algorithm, iou value
    print(f"Number of slices found: {len(dat)}") # == 33
    print(f"Number of neurons on slice 0: {dat[0].shape[0]}") # == number of ground truth neurons on slice 0

    # Get data to plot
    all_ious = []
    all_ious_no_flyback = []
    mean_iou_per_slice = []
    num_neurons_per_slice = []

    for i, slice in enumerate(dat):

        this_slice_iou = []
        for neuron in slice:
            this_iou = neuron[2] # scalar
            if this_iou > 0.0:
                this_slice_iou.append(this_iou) # add scalar to list

        # Main summary per slice
        if len(this_slice_iou)>0:
            mean_iou_per_slice.append(np.mean(this_slice_iou))
        else:
            mean_iou_per_slice.append(0)

        # All ious
        num_neurons_per_slice.append(len(slice))
        all_ious.extend(this_slice_iou) # do NOT add list to list
        if i > 0:
            all_ious_no_flyback.extend(this_slice_iou)

    # TODO: calculate mean across slices with IoUs > 0 & without the flyback

    # TODO: make a heatmap of the meaned IoU per parameter

    # TODO: Plot the minimum IoU of each parameter set

    # Multiple Plots
    plt.figure()
    plt.plot(all_ious, 'o')
    plt.title('All ious')
    plt.savefig(os.path.join(dat_folder, "all_ious.png"))

    plt.figure()
    plt.plot(all_ious_no_flyback, 'o')
    plt.title('All ious no flyback')
    plt.savefig(os.path.join(dat_folder, "all_ious_no_flyback.png"))


    plt.figure()
    plt.plot(mean_iou_per_slice)
    plt.xlabel("slice")
    plt.title("Mean iou per slice")
    plt.savefig(os.path.join(dat_folder, "mean_ious_per_slice.png"))


    plt.figure()
    plt.plot(num_neurons_per_slice)
    plt.xlabel("slice")
    plt.title("Number of neurons identified")
    plt.savefig(os.path.join(dat_folder, "neurons_identified.png"))

    plt.show()

    # change folder back to parent

# TODO: separate out the flyback slice