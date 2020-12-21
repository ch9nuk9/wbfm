import numpy as np
from scipy.ndimage import center_of_mass
import math
import csv

def write_pointcloud_from_masks(mask, fname):
    """
    Uses the centroid of a binary mask to create a pointcloud

    Saves a text file with columns of XYZ
    """

    all_neurons = np.unique(mask)

    binarized = (mask > 0).astype('uint8')
    com = center_of_mass(binarized, mask, all_neurons)

    with open(fname, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        for m in com:
# Gives decent tracking:
#             if not math.isnan(m[0]):# and m[0] > 1.0:
#                 m = list(m)
#                 m[0] = m[0]/10 -1.0
#                 m[1] = m[1]/250 -1.0 # z dimension is much smaller
#                 m[2] = m[2]/250 -1.0# z dimension is much smaller

            if not math.isnan(m[0]) and m[0] > 1.0:
                factor = 2.0
                m = factor*np.array(m)
                m[0] = m[0]/100 -0.2
                m[1] = m[1]/250 -1.0
                m[2] = m[2]/250 -1.0
                w.writerow(m)
    return com
