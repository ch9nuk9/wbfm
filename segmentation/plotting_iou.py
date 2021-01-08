"""
Here we are plotting the IoU results
"""

import matplotlib.pyplot as plt
import pickle
import os

# Get a filename for the data
dat_folder = 'dat'
fname = 'list_of_ious.pickle'
fname = os.path.join(dat_folder, fname)

# Import the data
with open(fname,'r') as file:
    dat = pickle.load(file)

# Check the sizes and whatnot
# 3 columns: neuron index of gt, index of algorithm, iou value
print(f"Number of slices found: {len(dat)}") # == 33
print(f"Number of neurons on slice 0: {dat[0].shape}") # == number of ground truth neurons on slice 0

# Get data to plot
# TODO: separate out the flyback slice
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

    num_neurons_per_slice.append(len(slice))
    all_ious.extend(this_slice_iou) # do NOT add list to list
    if i > 0:
        all_ious_no_flyback.extend(this_slice_iou)
    mean_iou_per_slice.append(np.mean(this_slice_iou))

# Multiple Plots

plt.scatter(all_ious)
plt.title('All ious')

plt.scatter(all_ious_no_flyback)
plt.title('All ious no flyback')

plt.plot(mean_iou_per_slice)
plt.xlabel("slice")
plt.title("Mean iou per slice")

plt.plot(num_neurons_per_slice)
plt.xlabel("slice")
plt.title("Number of neurons identified")
