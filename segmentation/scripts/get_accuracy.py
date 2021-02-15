"""

Plots the accuracy results from all algorithms

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation.util.segmentation_accuracy as acc
import pandas as pd
import pickle
from collections import defaultdict
from timeit import default_timer as timer

start = timer()

print('Start of accuracy calculations')

# TODO save the overlap areas!! (and somehow the mask area too)

# iterate over all results and save the metrics in a dataframe! (or matrix, if too much)
gt_path = r'C:\Segmentation_working_area\stitched_3d_data\gt_stitched_3d.npy'

algo_dir = r'C:\Segmentation_working_area\stitched_3d_data'

files = [os.path.join(algo_dir, f.name) for f in os.scandir(algo_dir) if f.is_file() and 'gt' not in f.name]

acc_results = defaultdict(dict)

for i, file in enumerate(files):
    print(f'Next file: {file}')
    key = os.path.split(file)[1][:-4]
    acc_results[key] = acc.seg_accuracy(gt_path, file)

    f_timer = timer()
    print(f'File {i}, total {round((f_timer - start), 2)} seconds')

print('.....Saving!')

# save
sv_dir = r'C:\Segmentation_working_area\accuracy'
pkl_file = os.path.join(sv_dir, 'accuracy_results_with_areas_and_percentages.pickle')
with open(pkl_file, 'wb') as pkl:
    pickle.dump(acc_results, pkl)

print(f'Finish time {timer() - start}')
print('Done with the script')
