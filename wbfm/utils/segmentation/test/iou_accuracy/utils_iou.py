"""

"""

import numpy as np

import pickle
from contextlib import redirect_stdout


def calculate_iou(ground_truth_path, algorithm_results_path):
    """
    Calculates the IoUs of a given segmentation result (3d array!) versus the ground truth

    Parameters
    ----------
    ground_truth_path:      path to ground truth data (seg.npy file)
    algorithm_results_path:  path to cellpose results data (.npy file)

    Returns
    -------
    For now, a list of arrays containing GT-ID, CP-ID & Iou (%)

    """
    print('Start of IoU calculation')
    print('ground truth path: ', ground_truth_path, '\ncellpose results path: ', algorithm_results_path)

    # quick logging. All commandline output will be written to a file (log_path)
    # log_interim = algorithm_results_path.split('\\')
    # log_path = "C:" + os.path.join('\\', *log_interim[1:-1], 'testlog.log')     # when just using split then join, the first '\' after 'C:' will be omitted!! =Problem

    log_path = "./iou_log.log"

    with open(log_path, 'w') as log_file:
        # using redirect_stdout to redirect the output of terminal to a file
        with redirect_stdout(log_file):

            print('First try with iou as a function')

            # Paths to data. Change this to test locally on your machine
            gt_path = ground_truth_path  # r"C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy"
            cp_path = algorithm_results_path  # r"C:\Users\niklas.khoss\Desktop\cp_vol\3d_nuclei\np_masks_3D__diam-100_flow-40.npy"

            # load ground truth and cellpose numpy data
            ground_truth = np.load(gt_path, allow_pickle=True).item()  # load gt. Allow pickle, as it is pickled data
            # print(ground_truth.keys())
            ground_truth = ground_truth['masks']

            cp = np.load(cp_path, allow_pickle=True)  # load cp data

            print('cp shape: ', cp.shape, ' gt shape: ', ground_truth.shape)  # print shapes to compare and check

            # Get all IOUs across planes
            # looping over all planes within the ground truth and try to use corresponding planes in cellpose result.
            # Add failsaves for mismatching amount of planes

            if ground_truth.shape[0] == cp.shape[0]:
                # initialize list output of all arrays to be pickled and dumped
                list_of_ious = []

                for plane in range(len(ground_truth)):
                    print(' --- PLANE ', plane, ' ---')

                    # get masks out of ground_truth
                    gt_plane = ground_truth[plane]

                    # load actual cellpose results (3D!)
                    cp_plane = cp[plane]
                    # print(cp_plane.shape)

                    # some plotting of planes etc
                    # plt.figure()
                    # plt.subplot(1,2,1)
                    # plt.title('cellpose results')
                    # plt.imshow(cp_plane)
                    # plt.subplot(1,2,2)
                    # plt.title('ground truth')
                    # plt.imshow(gt_plane)
                    # plt.show()

                    # Get neurons
                    gt_list = np.unique(gt_plane)
                    # print('gt_list: ', gt_list)

                    # initialize match output. the ID of the best match will be saved
                    best_match = np.zeros((len(gt_list), 2))

                    # initialize IoU output
                    # ious: [gt ID, cp ID, IoU value (%)]
                    # e.g. [254, 45, 63.43]
                    ious = np.zeros((len(gt_list), 3))

                    # print('best match shape: ', best_match.shape)

                    for i, neuron in enumerate(gt_list):
                        print('... mask ', i)

                        mask = gt_plane == neuron

                        # Intersection: comparing with cp results;
                        intersection = cp_plane[mask]

                        # Overlap:
                        # find all neurons and by how much they overlap
                        values, counts = np.unique(intersection, return_counts=True)

                        # store the ID of the pixels with the largest overlap of the intersection
                        match_value = values[np.argmax(counts)]  # argmax returns the index of max value
                        best_match[i, 0] = neuron
                        best_match[i, 1] = values[np.argmax(counts)]  # match_value
                        ious[i, 1] = values[np.argmax(counts)]

                        # Divide matches (intersections) by unions to get full IOUs
                        # make an IF clause, if match_value = 0 (== 0, if there is no match!)
                        if match_value > 0:
                            # print('--- Match = %d in i = %d' % (match_value, i))

                            # save the IoU value for each match
                            ious[i, 0] = int(neuron)

                            # intersection area = amount of best-match pixels
                            area_intersect = np.max(counts)

                            # area of union; get the masks of gt and cp (cp: where cp == match_value)
                            area_match = cp_plane == match_value  # array of matched value in cp
                            area_union = np.add(area_match, mask)  # add arrays of mask and match
                            area_union = np.count_nonzero(area_union)  # count non-zero elements in summed array

                            # IoU calculation and saving in 'ious' for each match
                            if area_union > 0 and area_intersect > 0:
                                iou_interim = (area_intersect / area_union) * 100
                                ious[i, 2] = round(iou_interim, 2)
                            else:
                                # print('Area = 0 for i = ', i)
                                ious[i, 1] = 0

                    list_of_ious.append(ious)
                    print('IoUs: \n', ious)
                    # print('best match output: ', best_match)

                else:
                    print('ERROR: Shapes of cp/gt arrays do not match!')

            # TODO: save the IoUs as pandas dataframes!
            # Save the IOUs (possibly better to return the IoUs and save it outside)
            # pickle/dump the list 'list_of_ious' in an output folder

            output_path = r"./list_of_ious.pickle"
            with open(output_path, 'wb') as pickle_out:
                pickle.dump(list_of_ious, pickle_out)

            # end of program
            print('end')


def plot_iou(working_directory, output_directory='./dat/'):
    """
    Plots the following IoU results:
        - Average across slices within a dataset
        -
    Parameters
    ----------
    working_directory
    output_directory

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np
    import os

    # Parent folder. Maybe change all of this script into a function and then use arg1 as parent!
    # for now: get current working directory
    parent = working_directory
    dirs = [d.name for d in os.scandir(parent) if d.is_dir()]  # list of subdirectories

    # TODO: how to determine the heatmap matrix dimensions
    # there is now a metadata.csv file in the parent folder. It contains the dimensions of parameter testings
    # we have 9x8 parameters (pixels 5:1:12 & flow 0.4:0.05:0.8)
    means_across_parameters = np.zeros((9, 8))

    minima = []
    counter = 0

    # iterate over folders
    for folder in dirs:
        cwd = os.path.join(parent, folder)

        # check, if folder contains any pickle files
        flist = os.listdir(cwd)
        for f in flist:
            if f.endswith('.pickle'):
                break
        else:
            continue

        os.chdir(cwd)

        # Get a filename for the input and output data
        fname = 'list_of_ious.pickle'
        fname = os.path.join(cwd, fname)

        savename_prefix = cwd.split(os.path.sep)
        savename_prefix = savename_prefix[-1].split('-')
        savename_prefix = 'p' + str(f[1]) + '_f' + str(f[3]) + '_'

        # Import the data
        with open(fname, 'rb') as file:
            dat = pickle.load(file)

        # ! means are identical, therefore print a slice to compare results across parameters
        #     print(dat[16])

        # Check the sizes and whatnot
        # 3 columns: neuron index of gt, index of algorithm, iou value
        #     print(f"Number of slices found: {len(dat)}") # == 33
        #     print(f"Number of neurons on slice 0: {dat[0].shape[0]}") # == number of ground truth neurons on slice 0

        # Get data to plot
        all_ious = []
        all_ious_no_flyback = []
        mean_iou_per_slice = []
        num_neurons_per_slice = []

        for i, slice in enumerate(dat):

            this_slice_iou = []
            for neuron in slice:
                this_iou = neuron[2]  # scalar
                if this_iou > 0.0:
                    this_slice_iou.append(this_iou)  # add scalar to list

            # Main summary per slice
            if len(this_slice_iou) > 0:
                mean_iou_per_slice.append(np.mean(this_slice_iou))
            else:
                mean_iou_per_slice.append(0)

            # All ious
            num_neurons_per_slice.append(len(slice))
            all_ious.extend(this_slice_iou)  # do NOT add list to list
            if i > 0:
                all_ious_no_flyback.extend(this_slice_iou)

        # calculate mean across slices with IoUs > 0 & without the flyback AND if > 10% (non-trivial)
        non_triv_means = [m for m in all_ious[1:] if m > 10]
        mean_across_slices = round(np.mean(non_triv_means), 2)

        # add IoU values to matrix for heatmap
        row = counter % means_across_parameters.shape[0]
        column = counter // means_across_parameters.shape[0]
        means_across_parameters[row, column] = mean_across_slices

        # Add the minimum IoU of each parameter set
        min_iou = min(all_ious)  # needed to use that roundabout way
        minima.append(min_iou)

        plots = False

        if plots:
            # Scatterplot: All IoUs (per neuron) within one dataset
            plt.figure()
            plt.plot(all_ious, 'o')
            plt.title('All ious')
            plt.savefig(os.path.join(cwd, savename_prefix, "all_ious.png"))

            # Scatterplot: All IoUs (per neuron) across slices without flyback-slice within one dataset
            plt.figure()
            plt.plot(all_ious_no_flyback, 'o')
            plt.title('All ious no flyback')
            plt.savefig(os.path.join(cwd, savename_prefix, "all_ious_no_flyback.png"))

            # Mean of IoUs within one slice across one dataset
            plt.figure()
            plt.plot(mean_iou_per_slice)
            plt.xlabel("slice")
            plt.title("Mean iou per slice")
            plt.savefig(os.path.join(cwd, savename_prefix, "mean_ious_per_slice.png"))

            plt.figure()
            plt.plot(num_neurons_per_slice)
            plt.xlabel("slice")
            plt.title("Number of neurons identified")
            plt.savefig(os.path.join(cwd, savename_prefix, "neurons_identified.png"))

            plt.close('all')

        #     plt.show()

        # change folder back to parent
        os.chdir(parent)

        counter += 1

    # heatmap of means of all IoUs within one dataset across parameters
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.title('Mean IoUs across parameters')
    im = plt.imshow(means_across_parameters)
    # create labels for axes
    xlabels = [str(x) for x in range(5, 13)]
    ylabels = [str(y / 100) for y in range(40, 85, 5)]

    # set the ticks before labelling, otherwise ticks/labels will be moved!
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.ylabel('Flow threshold')
    plt.xlabel('Diameter [Pixels]')
    for i in np.arange(means_across_parameters.shape[0]):
        for j in np.arange(means_across_parameters.shape[1]):
            text = ax.text(j, i, means_across_parameters[i, j],
                           ha="center", va="center", color="k")

    plt.savefig(os.path.join(output_directory, 'heatmap_of_IoU_means.png'))

    # plot minima of IoUs across parameters
    plt.figure()
    plt.title('IoU minima across parameters')
    plt.plot(minima, 'o')
    plt.xlabel('parameter [a.u.]')
    plt.ylabel('IoU (%)')

    plt.show

    plt.savefig(os.path.join(output_directory, 'IoU_minima_across_parameters.png'))

    print('end of script')
