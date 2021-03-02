import os, importlib
import numpy as np
import matplotlib.pyplot as plt


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