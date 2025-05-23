import copy
import glob
import os
import re

import numpy as np
import open3d as o3d
# My imports
import pandas as pd
import scipy
# import transformations as trans
from probreg import bcpd, cpd
from probreg import callbacks
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.general.utils_features import build_neuron_tree
from wbfm.utils.general.point_clouds.utils_bcpd_segmentation import bcpd_to_pixels, pixels_to_bcpd


##
## Helper functions based on importing keypoints
##

def prepare_source_and_target_nonrigid_3d(source_filename,
                                          target_filename,
                                          voxel_size=5.0):
    """
    Import data from FIJI-created .csv files using pandas and normalize
    Output: 2 open3d PointCloud objects
    """
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    # Read a dataframe, not just a text file
    try:
        df1 = pd.read_csv(target_filename)
        df2 = pd.read_csv(source_filename)
        target_np = df1[['XM', 'YM', 'ZM']].to_numpy()
        source_np = df2[['XM', 'YM', 'ZM']].to_numpy()
        # Test: normalize each column
        # target_np = target_np / target_np.max(axis=0)
        # source_np = source_np / source_np.max(axis=0)
    except:
        source_np = np.loadtxt(source_filename)
        target_np = np.loadtxt(target_filename)

    source.points = o3d.utility.Vector3dVector(source_np)
    target.points = o3d.utility.Vector3dVector(target_np)
    if voxel_size is not None:
        source = source.voxel_down_sample(voxel_size=voxel_size)
        target = target.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    print(target)
    return source, target


def correspondence_from_transform(tf_param, source, target):
    """
    Requires bcpd registration

    From the learned registration and the source and target distributions,
    calculate the closest neighbor to determine correspondence

    Uses sklearn.neighbors
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3d.geometry.PointCloud) else x)

    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    target_loc, result_loc = cv(target), cv(result)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_loc)
    distances, indices = nbrs.kneighbors(result_loc)
    # print("Fitting {} transformed points to {} target points".format(len(result_loc), len(target_loc)))

    return indices, distances


##
## Full matching traces
##

def match_2vol_BCPD(neurons0, neurons1,
                    w=0.0,
                    bcpd_kwargs={},
                    do_zscore=False,
                    do_any_preprocessing=True,
                    voxel_size=None,
                    DEBUG=False):
    """
    Matches using Bayesian Coherent Point drift

    VERY sensitive to settings:
        w - Weight of the uniform distribution in matching
        bcpd_kwargs = {'lmd':2.0, 'k':1.0e20, 'gamma':1.0}
            Lambda. Positive. It controls the expected length of displacement vectors.
            Kappa. Positive. It controls the randomness of mixing coefficients.
            Gamma. Positive. It defines the randomness of the point matching during the early stage of the optimization.

    See also: https://github.com/ohirose/bcpd#tuning-parameters
    """

    # Build pointclouds with normalized coordinates
    options = {'to_mirror': False}
    if do_any_preprocessing:
        if not do_zscore:
            f = lambda this_n: pixels_to_bcpd(np.array([np.array(n) for n in this_n]))
        else:
            f = lambda this_n: scipy.stats.zscore(np.array([np.array(n) for n in this_n]))
    else:
        f = lambda x: x
    _, pc0, _ = build_neuron_tree(f(neurons0), **options)
    _, pc1, _ = build_neuron_tree(f(neurons1), **options)
    if DEBUG:
        pc0.paint_uniform_color([0.5, 0.5, 0.5])
        pc1.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pc0, pc1])
        # print("Points of pc0: ", np.asarray(pc0.points))

    if voxel_size is not None:
        pc0 = pc0.voxel_down_sample(voxel_size=voxel_size)
        pc1 = pc1.voxel_down_sample(voxel_size=voxel_size)
    # Do BCPD
    options = {'w': w, 'maxiter': 500, 'tol': 1e-8}
    tf_param = bcpd.registration_bcpd(pc0, pc1, **options, **bcpd_kwargs)

    ## Convert into pairwise matches
    all_matches, all_conf = correspondence_from_transform(tf_param, pc0, pc1)
    all_matches = np.array([[i, val[0]] for i, val in enumerate(all_matches)])

    return all_matches, all_conf


def match_2vol_rigid(neurons0, neurons1,
                     w=0.0,
                     do_zscore=False,
                     voxel_size=None,
                     tf_type_name='rigid',
                     do_any_preprocessing=True,
                     DEBUG=False):
    """
    Matches using RIGID Coherent Point drift

    See also: match_2vol_BCPD
    """

    # Build pointclouds with normalized coordinates
    options = {'to_mirror': False}
    if do_any_preprocessing:
        if not do_zscore:
            f = lambda this_n: pixels_to_bcpd(np.array([np.array(n) for n in this_n]))
        else:
            f = lambda this_n: scipy.stats.zscore(np.array([np.array(n) for n in this_n]))
    else:
        f = lambda x: x
    _, pc0, _ = build_neuron_tree(f(neurons0), **options)
    _, pc1, _ = build_neuron_tree(f(neurons1), **options)

    if voxel_size is not None:
        pc0 = pc0.voxel_down_sample(voxel_size=voxel_size)
        pc1 = pc1.voxel_down_sample(voxel_size=voxel_size)
    # See: https://github.com/neka-nat/probreg/blob/master/examples/cpd_rigid_cuda.py

    # compute cpd registration
    tf_param, _, _ = cpd.registration_cpd(pc0, pc1,
                                          tol=1e-8,
                                          tf_type_name=tf_type_name)

    if DEBUG:
        result = copy.deepcopy(pc0)
        result.points = tf_param.transform(result.points)

        # draw result
        pc0.paint_uniform_color([0.5, 0.5, 0.5])
        pc1.paint_uniform_color([0, 0, 0])
        result.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([pc0, pc1, result])

    ## Convert into pairwise matches
    all_matches, all_conf = correspondence_from_transform(tf_param, pc0, pc1)
    all_matches = np.array([[i, val[0]] for i, val in enumerate(all_matches)])

    return all_matches, all_conf


##
## For saving the output
##

def save_indices(indices, fname=None):
    """ Saves indices (csv) using a standard format """

    df = pd.DataFrame({'indices': np.ndarray.flatten(indices)})
    # df = pd.DataFrame.from_records({'indices':indices})
    if fname is None:
        fname = 'test_bcpd_indices.csv'
    df.to_csv(fname, header=False)

    return fname


def save_indices_DLC(indices,
                     centroid_fnames,
                     centroid_format='DLC',
                     fname=None,
                     scorer='Charlie'):
    """
    Saves indices in DeepLabCut format

    centroid_fnames should be a .csv of the segmentations

    """

    num_neurons, num_tracked_pairs = np.shape(indices)

    # Original centroid statistics have the XYZ locations
    centroid_dfs = []
    centroid_nums = []
    for fname in centroid_fnames:
        # Get the number in the filename
        this_num = re.findall(r'\d+', fname)
        centroid_nums.append(int(this_num[0]))
        if centroid_format == 'DLC':
            centroid_dfs.append(pd.read_csv(fname))
        else:
            # names = ['X', 'Y', 'Z']
            names = ['Z', 'X', 'Y']
            centroid_dfs.append(pd.read_csv(fname, sep=' ', names=names))

    dataFrame = None
    coords = np.empty((len(centroid_dfs), 3,))

    output_path = ''
    # Get list of images
    #  Instead of looking at all image files, only parse the processed centroids
    # imlist=[]
    # imlist.extend([fn for fn in glob.glob(os.path.join(output_path,'*.csv')) if ('Statistics' in fn)])

    if len(centroid_fnames) == 0:
        print("No images found; aborting")
        return
    else:
        print("{} images found".format(len(centroid_fnames)), centroid_fnames)

    # index = np.sort(imlist)
    # print(index)
    print('Working on folder: {}'.format(os.path.split(str(output_path))[-1]))
    print("Note: this does not have the exact DLC format, but is specific to Linux")
    # Note: only works for single-digit indexed images

    # Define output for DLC on cluster
    subfoldername = 'test_100frames.ome'  # COMBAK
    # WARNING: hardcode linux filesep
    # Use the numbers from the centroid files
    print(centroid_nums)
    relativeimagenames = ['/'.join(('labeled-data', subfoldername, 'img{}.tif'.format(i))) for i in centroid_nums]
    print(relativeimagenames)

    # Build correctly DLC-formatted dataframe
    for i_neuron in range(num_neurons):
        bodypart = 'neuron{}'.format(i_neuron)

        # Get the index of THIS neuron in each file
        target_ind = [indices[i_neuron, i_target] for i_target in range(num_tracked_pairs)]
        ind_in_files = [i_neuron]
        ind_in_files.extend(target_ind)
        # ind_in_files = [i_neuron, indices[i_neuron][0]]
        # print("Tracked neuron from {} (source) to {} (target)".format(ind_in_files[0], ind_in_files[1]))

        # Get xyz coordinates for one neuron, for all files

        for i_source, df in enumerate(centroid_dfs):
            i_target = ind_in_files[i_source]  # The neuron index for this file
            x, y, z = [df['X'][i_target], df['Y'][i_target], df['Z'][i_target]]
            z, x, y = bcpd_to_pixels([z, x, y])  # Convert back to pixel space
            coords[i_source, :] = np.array([x, y, z])
            # print("Coordinates for neuron {}, file {}".format(i, i2), coords[i2,:])

        index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                            ['x', 'y', 'z']],
                                           names=['scorer', 'bodyparts', 'coords'])

        frame = pd.DataFrame(coords, columns=index, index=relativeimagenames)
        dataFrame = pd.concat([dataFrame, frame], axis=1)

    dataFrame.to_csv(os.path.join(output_path, "CollectedData_" + scorer + ".csv"))
    dataFrame.to_hdf(os.path.join(output_path, "CollectedData_" + scorer + '.h5'), 'df_with_missing', format='table',
                     mode='w')

    return output_path, dataFrame


##
## Initialize and align
##

if __name__ == "__main__":
    # Visualization
    to_plot = False

    # Get list of processed frames in folder
    output_path = ''
    # target_fnames=[]
    target_fnames = [fn for fn in glob.glob(os.path.join(output_path, '*.csv')) if ('Statistics' in fn)]

    # Choose a source frame; the rest are targets
    source_index = 3
    source_fname = target_fnames.pop(source_index)
    print("Source filename: ", source_fname)
    print("Target fnames: ", target_fnames)

    # Loop over the pairs
    all_indices = None
    for t_fname in target_fnames:
        # All the same source
        source, target = prepare_source_and_target_nonrigid_3d(source_fname,
                                                               t_fname,
                                                               0.005)
        if to_plot:
            cbs = [callbacks.Open3dVisualizerCallback(source, target)]
        else:
            cbs = []
        # Do BCPD
        print("Note: the value of w may need to be tweaked")
        tf_param = bcpd.registration_bcpd(source, target, w=1e-12,
                                          # gamma=10.0, #lmd=0.2, #k = 1e2,
                                          maxiter=100,
                                          callbacks=cbs)
        # Compute correspondence
        indices = correspondence_from_transform(tf_param, source, target)
        # save_indices(indices)
        if all_indices is None:
            all_indices = indices
        else:
            # print(all_indices.shape)
            # print(indices.shape)
            np.hstack((all_indices, indices))

    ## directly output in DLC format
    all_fnames = [source_fname].extend(target_fnames)
    print(all_fnames)
    save_indices_DLC(indices, all_fnames)
