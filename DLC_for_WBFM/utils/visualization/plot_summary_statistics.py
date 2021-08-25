import os

import numpy as np
import tifffile


def apply_function_to_pages(video_fname,
                            func,
                            num_pages=None):
    """
    Applies 'func' to each xy plane of an ome tiff, looping over pages

    This loop goes over pages, and makes sense when the metadata is corrupted

    """
    dat = []

    with tifffile.TiffFile(video_fname, multifile=False) as tif:
        if num_pages is None:
            num_pages = len(tif.pages)
        for i, page in enumerate(tif.pages):

            if i >= num_pages: break

            if i % 100 == 0: print(f'Page {i}/{num_pages}')
            dat.append(func(page.asarray()))

    return np.array(dat)
