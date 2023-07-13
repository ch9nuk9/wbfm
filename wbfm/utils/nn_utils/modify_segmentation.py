import numpy as np
import skimage


def remove_local_percentile_using_napari(viewer, sigma=1, percentile=10):

    t = viewer.dims.current_step[0]
    red_dat = np.array(viewer.layers['red_dat'].data[t, ...])
    seg_dat = np.array(viewer.layers['seg_dat'].data[t, ...])

    seg_dat, filtered_red = remove_local_percentile(percentile, red_dat, seg_dat, sigma)

    return seg_dat, filtered_red


def remove_local_percentile(percentile, red_dat, seg_dat, sigma):
    # Filter the brightness
    filtered_red = skimage.filters.gaussian(red_dat, sigma=sigma)
    print("Looping")
    ids = np.unique(seg_dat)[1:]
    for i in ids:
        this_mask = seg_dat == i
        # this_dat = np.where(, red_dat, 0)
        this_dat_no_zeros = filtered_red[this_mask]
        this_percentile = np.percentile(this_dat_no_zeros, percentile)
        to_set_to_zero = np.logical_and(this_mask, filtered_red < this_percentile)
        seg_dat[to_set_to_zero] = 0
    return seg_dat, filtered_red
