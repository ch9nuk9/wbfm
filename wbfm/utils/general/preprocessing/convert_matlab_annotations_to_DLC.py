import csv
import glob
import itertools
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions


def wb_tracker2dlc_format(path_config_file):
    """
    Converts Zimmer Whole-Brain tracker data for immobilized worms into training data for DeepLabCut
    """

    # Build filenames
    home = os.path.dirname(path_config_file)
    wb_fname = os.path.join(home, 'wbstruct.mat')

    config_file = Path(path_config_file).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    # Note: the labeled-data subfolder has the entire video name as the folder name
    video_fname = [i for i in cfg['video_sets'].keys()][0]  # Assume one video for now
    fname = Path(video_fname)
    output_path = os.path.join(Path(path_config_file).parents[0], 'labeled-data', fname.stem)
    output_path = output_path
    print('Looking in folder {}'.format(output_path))

    # Get list of images
    imlist = []
    imtype = '*.tif'
    imlist.extend([fn for fn in glob.glob(os.path.join(output_path, imtype)) if ('labeled.png' not in fn)])

    if len(imlist) == 0:
        print("No images found!!")

    index = np.sort(imlist)
    print('Working on folder: {}'.format(os.path.split(str(output_path))[-1]))
    relativeimagenames = ['labeled' + n.split('labeled')[1] for n in index]

    # Build dataset using pandas; copied from from labeling_toolbox.py
    scorer = cfg['scorer']
    with h5py.File(wb_fname, 'r') as mat:
        num_neurons = int(mat['simple']['nn'][0][0])
        x = mat['simple']['x'][:, 0]  # NOTE: a flip is required here; not sure why
        y = mat['simple']['y'][:, 0]
        z = mat['simple']['z'][:]
        coords = np.empty((len(index), 3,))
        dataFrame = None

        for i in range(num_neurons):
            bodypart = 'neuron{}'.format(i)

            # Note: this requires a flip in the y direction; not sure why
            x_sz = 272
            coords[:] = np.array([x_sz - x[i], y[i], z[i][0]])

            index = pd.MultiIndex.from_product([[scorer], [bodypart], ['x', 'y', 'z']],
                                               names=['scorer', 'bodyparts', 'coords'])
            # print(index)
            # print(coords)
            frame = pd.DataFrame(coords, columns=index, index=relativeimagenames)
            dataFrame = pd.concat([dataFrame, frame], axis=1)

    dataFrame.to_csv(os.path.join(output_path, "CollectedData_" + scorer + ".csv"))
    dataFrame.to_hdf(os.path.join(output_path, "CollectedData_" + scorer + '.h5'), 'df_with_missing', format='table',
                     mode='w')

    print('Finished')


def wb_tracker2config_names(path_config_file):
    """
    Automatically updates the config file with the proper number of neurons, and deletes any other default bodyparts.
    Only affects the "bodyparts" field
    """

    # Get number of neurons from wbstruct
    home = os.path.dirname(path_config_file)
    wb_fname = os.path.join(home, 'wbstruct.mat')
    with h5py.File(wb_fname, 'r') as mat:
        num_neurons = int(mat['simple']['nn'][0][0])

    # Read in entire config file into a list
    config_rows = []
    with open(path_config_file) as config:
        c_reader = csv.reader(config)  # , delimiter=' ')
        for row in c_reader:
            config_rows.append(row)

    ## Delete the current bodypart lines
    delete_these_rows = False
    config_rows_edit = config_rows.copy()
    for row in config_rows:
        if row == ['bodyparts:']:
            delete_these_rows = True  # Start deleting next row
        elif row == ['start: 0']:
            delete_these_rows = False  # Do not delete this row, or others
            break
        elif delete_these_rows == True:
            # Don't delete either of the two above, but only in between those rows
            config_rows_edit.remove(row)

    ## Add in the named neuron lines
    # Using "list slicing" https://www.geeksforgeeks.org/python-insert-list-in-another-list/
    new_names = [['- neuron{}'.format(i)] for i in range(num_neurons)]
    insert_index = config_rows_edit.index(['start: 0'])
    config_rows_edit[insert_index:insert_index] = new_names

    ## Write the file again
    if True:
        with open(path_config_file, 'w', newline='') as config:
            c_writer = csv.writer(config)
            for row in config_rows_edit:
                c_writer.writerow(row)
    # for row in config_rows_edit:
    # c_writer.writerow(row)
    #   print(row[:])


def csv_annotations2config_names(path_config_file,
                                 annotations_fname=None,
                                 num_dims=2,
                                 to_add_skeleton=True):
    """
    Automatically updates the config file with the proper number of neurons, and deletes any other default bodyparts.
    Only affects the "bodyparts" field
    """

    # Get number of neurons from annotations
    df = pd.read_hdf(annotations_fname)
    num_neurons = int(df.shape[1] / num_dims)
    print("Adding body part annotations for {} neurons".format(num_neurons))

    # Use DLC function to edit the file
    names = list(df.columns.get_level_values(1))
    # Remove repetitions
    updates = {"bodyparts": [names[i] for i in range(0, len(names), num_dims)]}
    auxiliaryfunctions.edit_config(path_config_file, updates)

    # Add a skeleton (fully connected)
    if to_add_skeleton:
        edge_iter = itertools.combinations(names, 2)
        updates = {"skeleton": [list(e) for e in edge_iter]}
        auxiliaryfunctions.edit_config(path_config_file, updates)

    print("Finished! Check the snakemake_config.yaml file to make sure the bodyparts are properly written")
