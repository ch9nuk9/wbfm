
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import pickle

log_path = r'C:\Users\niklas.khoss\Desktop\cp_vol\3d_nuclei\testlog.log'

with open(log_path, 'w') as log_file:
    # actual logging
    sys.stdout = log_file
    print('First try with iou as a function')

    # Paths to data. Change this to test locally on your machine
    gt_path = ground_truth_path  # r"C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy"
    cp_path = cellpose_results_path  # r"C:\Users\niklas.khoss\Desktop\cp_vol\3d_nuclei\np_masks_3D__diam-100_flow-40.npy"

    # load ground truth and cellpose numpy data
    ground_truth = np.load(gt_path, allow_pickle=True).item()  # load gt. Allow pickle, as it is pickled data
    # print(ground_truth.keys())
    ground_truth = ground_truth['masks']

    cp = np.load(cp_path, allow_pickle=True)  # load cp data

    print('cp shape: ', cp.shape, ' gt shape: ', ground_truth.shape)  # print shapes to compare and check

    # TODO 2. Get all IOUs across planes
    # looping over all planes within the ground truth and try to use corresponding planes in cellpose result.
    # Add failsaves for mismatching amount of planes

    if ground_truth.shape[0] == cp.shape[0]:
        # initialize list output of all arrays to be pickled and dumped
        list_of_ious = []

        for plane in range(len(ground_truth)):
            print(' --- PLANE ', plane, ' ---')

            # get masks out of ground_truth
            gt_5 = ground_truth[plane]

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
            # plt.imshow(gt_5)
            # plt.show()

            # Get neurons
            gt_list = np.unique(gt_5)
            # print('gt_list: ', gt_list)

            # initialize match output. the ID of the best match will be saved
            best_match = np.zeros((len(gt_list), 2))

            # initialize IoU output
            # ious: [gt ID, cp ID, IoU value (%)]
            # e.g. [254, 45, 63.43]
            ious = np.zeros((len(gt_list), 3))

            print('best match shape: ', best_match.shape)

            for i, neuron in enumerate(gt_list):
                print('... mask ', i)

                mask = gt_5 == neuron

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

                # TODO 1. Divide matches (intersections) by unions to get full IOUs
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
                        print('Area = 0 for i = ', i)
                        ious[i, 1] = 0

            list_of_ious.append(ious)
            print('IoUs: \n', ious)
            # print('best match output: ', best_match)

        else:
            print('ERROR: Shapes of cp/gt arrays do not match!')

    # TODO 3. Save the IOUs
    # pickle/dump the list 'list_of_ious' in an output folder
    output_path = r"C:\Users\niklas.khoss\Desktop\cp_vol\outputs\list_of_ious.pickle"
    with open(output_path, 'wb') as pickle_out:
        pickle.dump(list_of_ious, pickle_out)

    # end of program
    print('end')

    # TODO 4. Get all IOUs across parameters and save
    # Maybe make this script a function, which takes (at least) 2 inputs: gt & cp paths, and then runs this whole show.
    # The new function should iterate over folder contents of the cp-results.
    # It should write the output for each cp-file and maybe compare across
    #

    # TODO 5. Plot
    #
