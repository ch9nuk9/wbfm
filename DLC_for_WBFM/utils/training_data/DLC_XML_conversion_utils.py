from lxml import etree as ET
import pandas as pd
import os
import pathlib
import numpy as np
import deeplabcut
import csv

##
## Convert from Icy .xml files to DLC .h5 and .csv files
##


def icy_xml_to_dlc(path_config_file,
                   include_which_z_slices=None,
                   icy_annotation_fname=None,
                   using_original_fnames=False,
                   save_z_coordinate=False,
                   dlc_template_fname=None):
    """
    Converts 3d annotations done with the Icy GUI to DLC format

    Parameters
    ----------
    path_config_file : Path to config file of the project

    icy_annotation_fname : Icy annotations; by default, searches in the project

    Examples
    --------
    """

    # Settings
    if save_z_coordinate:
        coord_names = ['x', 'y', 'z']
    else:
        coord_names = ['x', 'y']

    config_file = pathlib.Path(path_config_file).resolve()
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_file)
    scorer = cfg['scorer']

    # Read in the template DLC tracks
    if dlc_template_fname is not None:
        df_original = pd.read_hdf(dlc_template_fname)
        df_original = df_original.sort_index()

        scorer = df_original.columns.levels[0][0]

        relativeimagenames=df_original.index

    # Get folder with annotations
    project_folder, xml_folder, xml_filename, png_fnames = find_xml_in_project(path_config_file)
    icy_annotation_fname = os.path.join(xml_folder, xml_filename)

    # Import XML
    # todo: detect from the config file?
    et_icy = ET.parse(icy_annotation_fname)
    et2 = et_icy.getroot()
    num_trackgroups = len(et2) - 2
    print("Found {} group(s) of tracks".format(num_trackgroups))

    # Write dataframe in DLC format
    if not using_original_fnames:
        relativeimagenames = []
        xml_name = os.path.basename(xml_folder)
        for f in png_fnames:
            relativeimagenames.append(os.path.sep.join(['labeled-data',xml_name,f]))
        # relativeimagenames = ['/'.join((folder_1, folder_2, fname_template.format(i))).replace(' ', '0') for i in range(num_files)]
    print("Relative image names:")
    print(relativeimagenames)

    dataFrame = None
    i_neuron_name = 0
    # Build correctly DLC-formatted dataframe
    for i_trackgroup in range(num_trackgroups):

        i_xml = i_trackgroup + 1 # The first entry in the xml file is the 'trackfile' class
        for this_detection in et2[i_xml]:
            bodypart = 'neuron{}'.format(i_neuron_name)
            frame = add_detection_to_df(this_detection,
                                        relativeimagenames, save_z_coordinate,
                                        scorer, bodypart, coord_names)
            if frame is not None:
                dataFrame = pd.concat([dataFrame, frame],axis=1)
                i_neuron_name = i_neuron_name + 1

    # Last: save
    # scorer = "Charlie"
    # scorer = "test"
    output_path = os.path.join(project_folder, 'labeled-data', xml_folder)
    dataFrame.to_csv(os.path.join(output_path,"CollectedData_" + scorer + ".csv"))
    deeplabcut.convertcsv2h5(path_config_file, userfeedback=False)
    # dataFrame.to_hdf(os.path.join(output_path,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

    print(f"Finished writing .csv and .h5; wrote {i_neuron_name} neurons")

    csv_annotations2config_names(path_config_file, len(coord_names))
    print("Finished updating config file with")

    return dataFrame


##
## Helper functions
##

def add_detection_to_df(this_detections,
                        relativeimagenames, save_z_coordinate,
                        scorer, bodypart, coord_names):
    # Get xyz or xy coordinates for one neuron, for all files
    coords = np.empty((len(relativeimagenames),len(coord_names),))
    for i2 in range(len(relativeimagenames)):
        try:
            this_track = this_detections[i2]
        except:
            print("Track not long enough; skipping: ", bodypart)
            return
        # COMBAK: Check whether the annotation is too far off the tracking slice
        # this_z = int(float(this_track.get('z')))
        # if include_which_z_slices is not None and this_z not in include_which_z_slices:
        #     continue
        if save_z_coordinate:
            coords[i2,:] = np.array([int(float(this_track.get('x'))),
                                     int(float(this_track.get('y'))),
                                     this_z ])
        else:
            coords[i2,:] = np.array([int(float(this_track.get('x'))),
                                     int(float(this_track.get('y')))])

    # Then, append to the dataframe (write at the end)
    index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                        coord_names],
                                        names=['scorer', 'bodyparts', 'coords'])

    frame = pd.DataFrame(coords, columns = index, index = relativeimagenames)

    return frame


def find_xml_in_project(path_config_file):
    project_folder = pathlib.Path(path_config_file).parent
    all_labeled_folders = os.listdir(os.path.join(project_folder, 'labeled-data'))

    for folder in all_labeled_folders:
        xml_folder = os.path.join(project_folder, 'labeled-data', folder)
        all_files = os.listdir(xml_folder)

        xml_fname = None
        png_fnames = []
        for f in all_files:
            if '.xml' in f:
                xml_fname = f # Assume only one
            elif '.png' in f:
                png_fnames.append(f)

        if xml_fname is not None:
            print(f"Found .xml annotations: {xml_fname}")
            break

    return project_folder, xml_folder, xml_fname, png_fnames


##
## Synchronizing the config file
##

# def csv_annotations2config_names(path_config_file, num_dims=2, actually_write=True):
#     """
#     Automatically updates the config file with the proper number of neurons, and deletes any other default bodyparts.
#     Only affects the "bodyparts" field
#     """
#
#     # Get number of neurons from annotations
#     home = os.path.dirname(path_config_file)
#     # COMBAK: must have XML in the folder to find it
#     annotations_fname = get_annotations_converted_from_xml(path_config_file)
#     # COMBAK: hardcoded folder
#     # annotations_fname = os.path.join(home,'labeled-data', 'test_100frames.ome','CollectedData_Charlie.csv')
#     df = pd.read_csv(annotations_fname)
#     num_neurons = int(df.shape[1] / num_dims)
#     print(f"Adding body part annotations for {num_neurons} neurons")
#     print("(Note: postprocessing may filter these tracks and only display a subset)")
#
#     # Read in entire config file into a list
#     config_rows = []
#     with open(path_config_file) as config:
#         c_reader = csv.reader(config)#, delimiter=' ')
#         for row in c_reader:
#             config_rows.append(row)
#
#     ## Delete the current bodypart lines
#     delete_these_rows = False
#     config_rows_edit = config_rows.copy()
#     for row in config_rows:
#         if row == ['bodyparts:']:
#             delete_these_rows = True # Start deleting next row
#         elif row == ['start: 0']:
#             delete_these_rows = False # Do not delete this row, or others
#             break
#         elif delete_these_rows == True:
#             # Don't delete either of the two above, but only in between those rows
#             config_rows_edit.remove(row)
#
#     ## Add in the named neuron lines
#     # Using "list slicing" https://www.geeksforgeeks.org/python-insert-list-in-another-list/
#     new_names = [['- neuron{}'.format(i)] for i in range(num_neurons)]
#     insert_index = config_rows_edit.index(['start: 0'])
#     config_rows_edit[insert_index:insert_index] = new_names
#
#     ## Write the file again
#     if actually_write:
#         with open(path_config_file, 'w', newline='') as config:
#             c_writer = csv.writer(config)
#             for row in config_rows_edit:
#                 c_writer.writerow(row)
#
#     print("Finished! Check the config.yaml file to make sure the bodyparts are properly written")


def get_annotations_converted_from_xml(path_config_file):

    project_folder, xml_folder, _, _ = find_xml_in_project(path_config_file)
    config_file = pathlib.Path(path_config_file).resolve()
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_file)
    scorer = cfg['scorer']
    output_path = os.path.join(project_folder, 'labeled-data', xml_folder)

    return os.path.join(output_path,"CollectedData_" + scorer + ".csv")
