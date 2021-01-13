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
                m = pixels_to_bcpd(m)
                w.writerow(m)
    return com


##
## Translations between the region that works for bcpd and pixel space
##

## Note: hardcoded to be inversions of each other...
def pixels_to_bcpd(m):
    factor = 2.0
    m = factor*np.array(m)
    m[0] = m[0]/100 - 0.2
    m[1] = m[1]/250 - 1.0
    m[2] = m[2]/250 - 1.0

    return m

def bcpd_to_pixels(m):
    factor = 2.0
    m = np.array(m)
    m[0] = (m[0]+0.2)*100/factor
    m[1] = (m[1]+1.0)*250/factor
    m[2] = (m[2]+1.0)*250/factor

    return m
