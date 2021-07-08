import csv
import h5py
import pandas as pd
import os

def bpcd_tracker2config_names(path_config_file):
    """
    Automatically updates the config file with the proper number of neurons, and deletes any other default bodyparts.
    Only affects the "bodyparts" field
    """

    # Get number of neurons from annotations
    home = os.path.dirname(path_config_file)
    # COMBAK: hardcoded folder
    annotations_fname = os.path.join(home,'labeled-data', 'test_100frames.ome','CollectedData_Charlie.csv')
    df = pd.read_csv(annotations_fname)
    num_neurons = int(df.shape[1] / 3)
    print("Adding body part annotations for {} neurons".format(num_neurons))
#     error()

    # Read in entire config file into a list
    config_rows = []
    with open(path_config_file) as config:
        c_reader = csv.reader(config)#, delimiter=' ')
        for row in c_reader:
            config_rows.append(row)

    ## Delete the current bodypart lines
    delete_these_rows = False
    config_rows_edit = config_rows.copy()
    for row in config_rows:
        if row == ['bodyparts:']:
            delete_these_rows = True # Start deleting next row
        elif row == ['start: 0']:
            delete_these_rows = False # Do not delete this row, or others
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

    print("Finished! Check the config.yaml file to make sure the bodyparts are properly written")
