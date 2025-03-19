"""

Plots the accuracy results from all algorithm results
(found in C:\Segmentation_working_area\data\bipartite_stitched_masks_3d).


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import wbfm.utils.segmentation.util.utils_accuracy as acc
import pandas as pd
import pickle
from collections import defaultdict
from timeit import default_timer as timer

start = timer()

print('Start of accuracy calculations')

# iterate over all results and save the metrics in a dataframe! (or matrix, if too much)
gt_path = r'C:\Segmentation_working_area\ground_truth\bipartite_stitched_gt\prealigned_gt_stitched\prealigned_gt_stitched_no_filter.npy'
# gt_path = r'C:\Segmentation_working_area\ground_truth\bipartite_stitched_gt\normal_gt_stitched\gt_bipartite_stitching.npy'

# algo_dir = r'C:\Segmentation_working_area\data\stitched\prealigned_stitched'
# algo_dir = r'C:\Segmentation_working_area\data\stitched\raw_vol_stitched'
# algo_dir = r'C:\Segmentation_working_area\data\leifer_segmentation'
algo_dir = r'C:\Segmentation_working_area\data\lukas_model\prealigned_no_filter'

files = [os.path.join(algo_dir, f.name) for f in os.scandir(algo_dir) if f.is_file() and
         'gt' not in f.name and f.name.endswith('.npy')]

acc_results = defaultdict(dict)


for i, file in enumerate(files):
    print(f'----- Next file: {file} ------')

    # change name here
    key = os.path.split(file)[1][:-4]
    acc_results[key] = acc.seg_accuracy(gt_path, file)

    f_timer = timer()
    print(f'File {i}, total {round((f_timer - start), 2)} seconds')

print('.....Saving!')

# save
sv_dir = r'C:\Segmentation_working_area\data\lukas_model\prealigned_no_filter'
pkl_file = os.path.join(sv_dir, 'lukas_prealigned_volume_accuracy_results.pickle')
with open(pkl_file, 'wb') as pkl:
    pickle.dump(acc_results, pkl)

print(f'Finish time {timer() - start}')
print('Done with the script')
