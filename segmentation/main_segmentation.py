"""

Main function of segmenting volumes. It generates predicted masks of volumes, which have
been given as input.

Input: filepath
Output: 3D mask segmented (by StarDist)

"""

from segmentation.util.stardist_seg import seg_with_stardist

def main_segmentation_sd(volume_path):

    # Read in volume

    # optional prealignment

    # segment with StarDist

    #

    return