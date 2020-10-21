from lxml import etree as ET
import pandas as pd



##
## Convert from Icy .xml files to DLC .h5 and .csv files
##


def icy_xml_to_dlc(dlc_template_fname,
                   icy_annotation_fname,
                   using_original_fnames=False,
                   save_z_coordinate=False,
                   output_path='.'):
    """
    Converts 3d annotations done with the Icy GUI to DLC format

    Parameters
    ----------
    dlc_template_fname : Template with the CORRECT FILE NAMES

    Examples
    --------
    """

    # Settings
    if save_z_coordinate:
        coord_names = ['x', 'y', 'z']
    else:
        coord_names = ['x', 'y']

    # Read in the template DLC tracks
    df_original = pd.read_hdf(dlc_fname)
    df_original = df_original.sort_index()

    all_files = df_original.index
    scorer = df_original.columns.levels[0][0]

    # Import XML
    # TODO: detect from the config file?
    et_icy = ET.parse(icy_annotation_fname)
    et2 = et_icy.getroot()
    num_trackgroups = len(et2) - 2
    print("Found {} group(s) of tracks".format(num_trackgroups))

    # Write dataframe in DLC format
    if using_original_fnames:
        relativeimagenames=df_original.index
    else:
        folder_1 = 'labeled-data'
        folder_2 = 'test_1000frames_13slice'
        fname_template = 'img{:2d}.png'
        num_files = 25
        relativeimagenames = ['/'.join((folder_1, folder_2, fname_template.format(i))).replace(' ', '0') for i in range(num_files)]
    print(relativeimagenames)
    print("Assumes filenames in the DLC annotation are same as the Icy tracker, after alphabetizing")

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
    dataFrame.to_csv(os.path.join(output_path,"CollectedData_" + scorer + ".csv"))
    dataFrame.to_hdf(os.path.join(output_path,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

    print(f"Finished; wrote {i_neuron_name} neurons")


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
        if save_z_coordinate:
            coords[i2,:] = np.array([int(float(this_track.get('x'))),
                                     int(float(this_track.get('y'))),
                                     int(float(this_track.get('z'))) ])
        else:
            coords[i2,:] = np.array([int(float(this_track.get('x'))),
                                     int(float(this_track.get('y')))])

    # Then, append to the dataframe (write at the end)
    index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                        coord_names],
                                        names=['scorer', 'bodyparts', 'coords'])

    frame = pd.DataFrame(coords, columns = index, index = relativeimagenames)

    return frame
