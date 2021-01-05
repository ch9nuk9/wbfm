"""

IoU (Intersection over Union) to be calculated from masks

"""
print('start')

import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Plane of choice while not looping over all planes etc
plane = 9

# Paths to data. Change this to test locally on your machine
gt_path = r"C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy"
cp_path = r"C:\Users\niklas.khoss\Desktop\cp_vol\3d_nuclei\np_masks_3D__diam-100_flow-40.npy"

# load ground truth numpy data
ground_truth = np.load(gt_path, allow_pickle=True).item()
print(ground_truth.keys())

# get masks out of ground_truth
gt_5 = ground_truth['masks'][plane]

# load actual cellpose results (3D!)
cp_5 = np.load(cp_path, allow_pickle=True)[plane]
print(cp_5.shape)

# some plotting of planes etc
plt.figure()
plt.subplot(1,2,1)
plt.title('cellpose results')
plt.imshow(cp_5)
plt.subplot(1,2,2)
plt.title('ground truth')
plt.imshow(gt_5)
plt.show()

# Get neurons
gt_list = np.unique(gt_5)
print('gt_list: ', gt_list)

# initialize match output. the ID of the best match will be saved
best_match = np.zeros((len(gt_list), 2))
# initialize IoU output
ious = np.zeros((len(gt_list), 2))

print('best match shape: ', best_match.shape)

for i, neuron in enumerate(gt_list):

    mask = gt_5 == neuron

    # Intersection: comparing with cp results;
    intersection = cp_5[mask]

    # Overlap:
    # find all neurons and by how much they overlap
    values, counts = np.unique(intersection, return_counts=True)

    # store the ID of the pixels with the largest overlap of the intersection
    match_value = values[np.argmax(counts)]        # argmax returns the index of max value
    best_match[i, 0] = neuron
    best_match[i, 1] = values[np.argmax(counts)]  # match_value

    # TODO 1. Divide matches (intersections) by unions to get full IOUs
    # make an IF clause, if match_value = 0 (== 0, if there is no match!)
    if match_value > 0:
        # print('--- Match = %d in i = %d' % (match_value, i))

        # save the IoU value for each match
        ious[i, 0] = int(neuron)

        # intersection area = amount of best-match pixels
        area_intersect = np.max(counts)

        # area of union; get the masks of gt and cp (cp: where cp == match_value)
        area_match = cp_5 == match_value                # array of matched value in cp
        area_union = np.add(area_match, mask)           # add arrays of mask and match
        area_union = np.count_nonzero(area_union)       # count non-zero elements in summed array

        # IoU calculation and saving in 'ious' for each match
        if area_union > 0 and area_intersect > 0:
            iou = (area_intersect / area_union) * 100
            ious[i, 1] = round(iou, 2)
        else:
            print('Area = 0 for i = ', i)
            ious[i, 1] = 0



print('IoUs: ', ious)
print('best match output: ', best_match)

# TODO 2. Get all IOUs across planes
# TODO 3. Save the IOUs
# TODO 4. Get all IOUs across parameters and save
# TODO 5. Plot

# end of program
print('end')
