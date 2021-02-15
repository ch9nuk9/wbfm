import numpy as np
import os
from collections import defaultdict
import segmentation.util.overlap as ol
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

def create_2d_masks_gt():
    # creates separate 2d masks of the annotated ground truth
    path = r'C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy'
    arr = np.load(path, allow_pickle=True).item()
    masks = arr['masks']

    sv_path = r'C:\Users\niklas.khoss\Desktop\cp_vol\gt_masks_npy'
    c = 0
    for m in masks[1:]:
        np.save(os.path.join(sv_path, 'gt_mask_' + str(c)), m, allow_pickle=True)
        print(os.path.join(sv_path, 'gt_mask_' + str(c)))
        c += 1

# create_2d_masks_gt()


# refactor into using 3d arrays instead of 2d
def gmm():
    # create a data -> brightness from a single neuron
    # data needed: original 3d array, stitched array of choice, neuron lengths

    img_data_path = r'C:\Segmentation_working_area\test_volume'
    og_3d = ol.create_3d_array_from_tiff(img_data_path)

    # stitched array (Stardist)
    sd_path = r'C:\Segmentation_working_area\stardist_testdata\masks'
    sd_3d = ol.create_3d_array(sd_path)
    sd_stitch, sd_nlen = ol.calc_all_overlaps(sd_3d)
    sd_bright = ol.calc_brightness(og_3d, sd_stitch, sd_nlen)

    # choose example neuron with z>20;
    # N > 16: [1, 13, 20, 23, 26, 29, 37, 39, 45, 47, 49, 52, 67]
    example_dist = sd_bright['1']
    # reshape distribution
    example_dist = example_dist.reshape(1, -1)

    gmm = GMM(n_components=2)

    model = gmm.fit(example_dist)
    x = np.linspace(1, len(example_dist), len(example_dist))


    print('Done with GMM')
    return

def what_is_x_when_y_is(input, x, y):
    order = y.argsort()
    x = x[order]
    y = y[order]

    # finds closest x-index of y-value coming from the left side
    return x[y.searchsorted(input, 'left')]

def brightness_histograms(brightness_dict: dict):
    plt.figure()
    # make a histogram for every entry in the brightness dict
    for k, v in brightness_dict.items():
        # x_means = []
        # try:
        #     x_means, g1, g2 = ol.calc_means_via_brightnesses(v, 0)
        # except RuntimeError:
        #     print(f'could not fit neuron: {k}')

        x_lin = np.arange(1, len(v) + 1)

        plt.plot(x_lin, v, color='b', label='Data')

        # if x_means:
        #     plt.plot(x_lin, g1, color='g', label='Fit1')
        #     plt.plot(x_lin, g2, color='r', label='Fit2')

        plt.title('Neuron: ' + str(k) + ' brightness')
        plt.ylabel('avg. brightness')
        plt.xlabel('Slice (relative)')

        sv_nm = r'C:\Segmentation_working_area\brightnesses'
        sv_nm = os.path.join(sv_nm, 'gt_brightness_after_stitching_' + str(k) + '.png')
        plt.tight_layout()
        plt.savefig(sv_nm)

    return


# gmm()