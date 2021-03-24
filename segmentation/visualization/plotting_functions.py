import os, importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_area_histograms(array_path, show_plots=0):

    array = np.load(array_path)
    areas = {}
    all_areas = []

    for i, p in enumerate(array):
        print(f'Slice: {i}')
        uniq = np.unique(p)
        inter_areas = []
        for u in uniq:
            if u == 0:
                continue
            areas[u] = np.count_nonzero(p == u)
            inter_areas.append(np.count_nonzero(p == u))

        all_areas.extend(inter_areas)

        plt.figure()
        plt.hist(areas.values(), bins=25)
        plt.title(f'Areas on plane {i}')
        sv1 = f'areas_{i}_' + os.path.split(array_path)[1][:-4]
        savename = os.path.join(r'C:\Segmentation_working_area\results\areas_per_slice', sv1)
        plt.savefig(savename)

        if show_plots >= 1:
            plt.show()

    print('Done with area histograms')
    

def plot_brightnesses(brightnesses, brightness_planes, save_path, save_flag=0):
    """
    plots the brightness histogram for all entries in the given dict.

    Parameters
    ----------
    brightnesses : dict
        {neuron id = [list of average brightnesses across Z]}
    brightness_planes : dict
        {neuron id = [list of Z planes with neuron in it]}
    save_path : str
        path for saving files
    save_flag : int
        flag for saving the plots in 'save_path'

    Returns
    -------

    """
    for k, v in brightnesses.items():
        if v:
            x = brightness_planes[k]
            fig = plt.figure(figsize=(1920/96, 1080/96), dpi=96)
            plt.plot(x, v)
            
            plt.xlabel('slice [Z]')
            plt.ylabel('Average brightness')
            
            tit_str = 'Neuron ' + str(k) + ' brightness'
            plt.title(tit_str)

            if save_flag > 0:
                plt.savefig(os.path.join(save_path + 'n' + str(k) + '_brightness_histogram.png'), dpi=96)
            
    return


def plot_volumes_of_overlaps(acc_results_pickle):
    # plot absolute and %-GT volumes
    # acc_res = r'C:\Segmentation_working_area\results\new_acc_metrics\all_accuracy_results_with_volumes_and_percentages.pickle'

    gt = np.load(r'C:\Segmentation_working_area\ground_truth\bipartite_stitched_gt\prealigned_gt_stitched\prealigned_gt_stitched_no_filter.npy')

    with open(acc_results_pickle, 'rb') as file:
        results = pickle.load(file)

        for k, v in results.items():
            vals = list()

            vols = results[k]['vol_gt']
            fig1 = plt.figure(figsize=(1920/96, 1080/96), dpi=96)

            print(f'{k} has {len(vols)} ')

            for key, val in vols.items():
                if v:
                    gt_vol = np.count_nonzero(gt == int(key))

                    vals.extend([round((x/gt_vol) * 100) for x in val])

            plt.hist(vals, bins=100)
            plt.xlim(0, 100)
            plt.xticks(list(range(0, 100, 10)), fontsize=18)
            plt.title('% volumes matching GT - ' + k, fontsize=24)
            plt.xlabel('% of GT volume matched by algorithm mask', fontsize=22)
            plt.ylabel('Occurrences', fontsize=22)
            plt.tight_layout()

            sv_path = os.path.join(os.path.split(acc_results_pickle)[0], 'volume_overlap_' + k)
            plt.savefig(sv_path, dpi=96)

            plt.close()
