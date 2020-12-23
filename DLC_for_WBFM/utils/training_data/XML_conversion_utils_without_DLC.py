from lxml import etree as ET
import pandas as pd
import os
import pathlib
import numpy as np
from DLC_for_WBFM.utils.point_clouds.utils_bcpd_segmentation import bcpd_to_pixels


##
## Other direction: convert dlc to Icy (for visualization)
##

def icy_xml_to_dlc(dlc_annotation_fname,
                   icy_annotation_fname=None,
                   experimenter=None):
    # Read the file
    df = pd.read_hdf(dlc_annotation_fname)
    if experimenter is None:
        experimenter = df.keys()[0][0]
    df = df[experimenter]

    # Sort
    df = df.sort_index() # Filenames may be a different order

    all_neurons = df.columns.levels[0]
    all_files = df.index

    # Make the actual XML
    # Initialize
    xm = ET.Element("root")
    xm1 = ET.SubElement(xm, "trackfile", version="1")

    # Add a neuron
    trackgroup = ET.SubElement(xm1, "trackgroup", description="All neurons from DLC")
    for i, this_neuron in enumerate(all_neurons):
        # trackgroup = ET.SubElement(xm1, "trackgroup", description="Neuron {} from DLC".format(i))
        t = ET.SubElement(trackgroup, "track", id=str(i))
        for i_t, this_file in enumerate(all_files):
            x = df[this_neuron]['x'][this_file]
            y = df[this_neuron]['y'][this_file]
            z = df[this_neuron]['z'][this_file]
            x, y, z = str(x), str(y), str(z)
            ET.SubElement(t, "detection",
                          classname="plugins.nchenouard.spot.Detection", #TODO
                          color=str(i),
                          t=str(i_t),
                          type="1",
                          x=y, y=x, z=z) # TODO: switch x and y

    # SAVE
    tree = ET.ElementTree(xm)

    if icy_annotation_fname is None:
        tmp = os.path.splitext(dlc_annotation_fname)[0]
        icy_annotation_fname = tmp + ".xml"

    tree.write(icy_annotation_fname,
               pretty_print=True,
               xml_declaration=False,
               with_tail=False,
               method="xml",
               inclusive_ns_prefixes=['Neuron 0 from DLC'])
