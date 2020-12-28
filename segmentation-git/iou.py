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

mask_5 = ground_truth['masks'][5]
# plt.figure()
# plt.imshow(mask_5)
# plt.show()

# load actual cellpose results
cp_path = r"C:\Users\niklas.khoss\Desktop\cp_vol\pixels-10-flow-0.60\np_masks_5_diam-100_flow-60.npy"
mask_cp_5 = np.load(cp_path, allow_pickle=True)
print(mask_cp_5.shape)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask_cp_5)
plt.subplot(1,2,2)
plt.imshow(mask_5)
plt.show()




print('end')
