"""

IoU (Intersection over Union) to be calculated from masks

"""
print('start')

import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load numpy data
path = r"C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy"
ground_truth = np.load(path, allow_pickle=True).item()

print(ground_truth.keys())

# get masks out of ground_truth

gt_5 = ground_truth['masks'][5]
# plt.figure()
# plt.imshow(gt_5)
# plt.show()

# load actual cellpose results
cp_path = r"C:\Users\niklas.khoss\Desktop\cp_vol\pixels-10-flow-0.60\np_masks_5_diam-100_flow-60.npy"
cp_5 = np.load(cp_path, allow_pickle=True)
print(cp_5.shape)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(cp_5)
# plt.subplot(1,2,2)
# plt.imshow(gt_5)
# plt.show()

# Get neurons
gt_list = np.unique(gt_5)
print('gt_list: ', gt_list)

# initialize match output
best_match = np.zeros((len(gt_list), 2))

print(best_match.shape)

for i, neuron in enumerate(gt_list):

    mask = gt_5 == neuron

    # Intersection: comparing with cp results;
    intersection = cp_5[mask]

    # Overlap:
    # find all neurons and by how much they overlap
    values, counts = np.unique(intersection, return_counts=True)

    best_match[i, 0] = neuron
    best_match[i, 1] = values[np.argmax(counts)]

print(best_match)



# end of program
print('end')
