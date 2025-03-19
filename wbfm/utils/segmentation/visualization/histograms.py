import os
import numpy as np
import matplotlib.pyplot as plt


def neuron_length_hist(lengths_dict: dict, save_path='', save_flag=0):
    # plots the lengths of neurons in a histogram and barplot
    vals = lengths_dict.values()
    keys = lengths_dict.keys()
    if save_path:
        fname = os.path.split(save_path)[1]
    else:
        fname=''

    fig = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.hist(vals, bins=np.arange(1, max(vals)+2), align='left')
    plt.xticks(np.arange(1, max(vals) + 2))
    # TODO add automated y-axis limits for histograms
    #plt.yticks(np.arange(0, 27, 2))
    plt.xlabel('neuron length')
    plt.ylabel('# of neurons')
    plt.title('neuron lengths histogram ' + fname)
    # plt.show()

    # TODO: change save folder
    if save_flag >= 1:
        plt.savefig(os.path.join(save_path, fname + '_neuron_lengths.png'), dpi=96)

    fig1 = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    plt.bar(keys, vals)
    plt.title('Neuron lengths per neuron ' + fname)
    plt.ylabel('Length')
    plt.xlabel('Neuron #')

    if save_flag >= 1:
        plt.savefig(os.path.join(save_path, fname + '_neuron_lengths_bar.png'), dpi=96)

    return fig, fig1


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