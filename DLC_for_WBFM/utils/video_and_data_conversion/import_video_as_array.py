import numpy as np
import tifffile
import os
from pathlib import Path
import cv2



def get_video_from_ome_file_subset(video_fname,
                                   num_frames = 100,
                                   frame_width = 608,
                                   frame_height = 610,
                                   num_slices=33,
                                   alpha=1.0):
    """
    Imports to np.array() from a single ome-tiff file that is incomplete, i.e. cannot be read using tifffile.imread()


    """

    # Data format: TZHW
    dat = np.zeros((num_frames, num_slices, frame_height, frame_width))

    i_max = (num_frames-1)*num_slices

    with tifffile.TiffFile(video_fname, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            if i%100==0:
                print(f'Page {i}/{i_max}')
            if i > i_max: break

            # These pages are a single z slice
            img = page.asarray()
            img = (alpha*img).astype('uint8')

            # Find the correct indices
            i_t = i // num_slices
            i_z = i % num_slices

            # Finally, save
            dat[i_t, i_z,...] = img

    return dat


def get_single_volume(fname, which_vol, num_slices, alpha=1.0, dtype='uint8'):
    # Convert to page coordinates
    start_ind = num_slices*which_vol
    key = range(start_ind, start_ind+num_slices)
    dat = (alpha*tifffile.imread(fname, key=key)).astype(dtype)

    return dat


def get_adjacent_volumes(fname, first_vol, num_slices):
    dat0 = get_single_volume(fname, first_vol, num_slices)
    dat1 = get_single_volume(fname, first_vol+1, num_slices)
    return dat0, dat1
