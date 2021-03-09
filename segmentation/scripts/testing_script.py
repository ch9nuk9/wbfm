import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation.util.overlap as ol
import pickle


sd_dir = r'C:\Segmentation_working_area\data\stardist_raw\3d'
sd_files = [os.path.join(sd_dir, f.name) for f in os.scandir(sd_dir) if f.is_file()]

cp_dir = r'C:\Segmentation_working_area\data\cellpose'
cp_files = [os.path.join(cp_dir, f.name) for f in os.scandir(cp_dir) if f.is_file()]

all_algo_files = cp_files + sd_files
# all_algo_files = sd_files

# load raw array
raw_array = ol.create_3d_array_from_tiff(
    r'C:\Segmentation_working_area\data\raw_volume')
if raw_array.shape[0] == 33:
    raw_array = raw_array[1:]

preprocessed_array = ol.create_3d_array_from_tiff(
    r'C:\Segmentation_working_area\data\raw_volume\preprocessed')
if preprocessed_array.shape[0] == 33:
    preprocessed_array = preprocessed_array[1:]

print('raw array shapes: ', raw_array.shape, preprocessed_array.shape)

for file in all_algo_files:
    print(f'File: {file}')
    algo_array = np.load(file)

    if algo_array.shape[0] == 33:
        print(f'{file} had 33 slices')
        algo_array = algo_array[1:]

    if 'pre' in file:
        final_mask, final_lengths, final_brightnesses, brightness_planes, df = ol.level2_overlap(preprocessed_array, algo_array)
        print(f'- preprocessed: {file}')
    else:
        final_mask, final_lengths, final_brightnesses, brightness_planes, df = ol.level2_overlap(raw_array, algo_array)

    # first, create results folder if not existent
    results_path = os.path.join(r'C:\Segmentation_working_area\results\accuracy_summary_results',
                                os.path.split(file)[1][:-4])
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    np.save(os.path.join(results_path, os.path.split(file)[1][:-4] + '_bipartite_stitched_mask_final'),
            final_mask, allow_pickle=True)

    with open(os.path.join(results_path, os.path.split(file)[1][:-4] + 'lengths_and_brightnesses.pickle'), 'wb') as pickle_out:
        pickle.dump([final_lengths, final_brightnesses, brightness_planes], pickle_out)

    # save length histograms
    h1, h2 = ol.neuron_length_hist(final_lengths, results_path, 1)

print('Completely done!')


