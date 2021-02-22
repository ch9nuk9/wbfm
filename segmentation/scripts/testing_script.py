import os, importlib
import segmentation.util.overlap as ol
import segmentation.util.under_construction as uc
import numpy as np
import matplotlib.pyplot as plt

gt_path = 'C:\\Segmentation_working_area\\testing_area\\gt'
gt_3d = ol.create_3d_array(gt_path)

sd_path = r'C:\Segmentation_working_area\testing_area\sd'
sd_3d = ol.create_3d_array(sd_path)

print(f' Stitching')
ol_gt_stitched, ol_gt_len = ol.calc_all_overlaps(gt_3d)

gt_3d = ol.create_3d_array(gt_path)
bp_gt_stitched = ol.bipartite_stitching(gt_3d)
bp_gt_len = ol.get_neuron_lengths_dict(bp_gt_stitched)

print(f'overlap lengths: {len(ol_gt_len)}  ---  bipartite lengths: {len(bp_gt_len)}')

print('Done')
