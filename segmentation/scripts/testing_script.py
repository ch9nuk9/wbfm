import os, importlib
import segmentation.util.overlap as ol
import segmentation.util.under_construction as uc
import numpy as np
import matplotlib.pyplot as plt

gt_path = 'C:\\Segmentation_working_area\\ground_truth\\gt_masks_npy'
gt_3d = ol.create_3d_array(gt_path)

print(f' Stitching')
gt_3d = uc.bipartite_stitching(gt_3d)

print('Done')