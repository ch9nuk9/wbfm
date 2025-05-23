import copy
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wbfm.utils.general.utils_features import build_neuron_tree
from wbfm.utils.neuron_matching.utils_keypoint_matching import get_indices_of_tracklet


def visualize_tracks(neurons0, neurons1, matches=None, trivial_matches=False, to_plot_failed_lines=False, to_plot=True):
    """

    Parameters
    ----------
    neurons0 - painted gray
    neurons1 - painted black
    matches - list of 2-element indices
    to_plot_failed_lines

    Returns
    -------

    """
    import open3d as o3d

    n0, n1, pc_n0, pc_n1 = build_pair_of_point_clouds(neurons0, neurons1)

    # Plot lines from initial neuron to target
    points = np.vstack((pc_n0.points, pc_n1.points))

    if trivial_matches:
        matches = [[i, i] for i in range(max(n0, n1))]

    if matches is not None:
        combined_matches = []
        for i, match in enumerate(matches):
            combined_matches.append([match[0], n0 + match[1]])

        successful_lines = []
        failed_lines = []
        for row in combined_matches:
            if row[1] != n0:
                successful_lines.append(row)
            else:
                failed_lines.append(row)

        successful_colors = [[0, 1, 0] for i in range(len(successful_lines))]
        successful_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(successful_lines),
        )
        successful_line_set.colors = o3d.utility.Vector3dVector(successful_colors)
        if to_plot_failed_lines:
            failed_colors = [[1, 0, 0] for i in range(len(failed_lines))]
            failed_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(failed_lines),
            )
            failed_line_set.colors = o3d.utility.Vector3dVector(failed_colors)
            to_draw = [failed_line_set, successful_line_set, pc_n0, pc_n1]
        else:
            to_draw = [successful_line_set, pc_n0, pc_n1]
    else:
        to_draw = [pc_n0, pc_n1]

    coords = build_coordinate_arrows()
    to_draw.append(coords)
    if to_plot:
        o3d.visualization.draw_geometries(to_draw)

    return to_draw


def build_coordinate_arrows(base=None):
    if base is None:
        base = np.array([-1, -1, -1])
    import open3d as o3d
    target_pts = base + np.array([[0, 0, 0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    matches = [[0, 1], [0, 2], [0, 3]]
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(target_pts),
        lines=o3d.utility.Vector2iVector(matches)
    )
    lines.colors = o3d.utility.Vector3dVector(target_colors)
    return lines


def build_pair_of_point_clouds(neurons0, neurons1):
    n0, pc_n0, tree_neurons0 = build_neuron_tree(neurons0, to_mirror=False)
    n1, pc_n1, tree_neurons1 = build_neuron_tree(neurons1, to_mirror=False)
    # pc_n0.paint_uniform_color([0.5, 0.5, 0.5])
    pc_n0.paint_uniform_color([1, 0, 0])
    pc_n1.paint_uniform_color([0, 0, 0])
    return n0, n1, pc_n0, pc_n1


def visualize_tracks_simple(pc0, pc1, matches):
    import open3d as o3d
    pc0.paint_uniform_color([1, 0, 0])
    pc1.paint_uniform_color([0, 1, 0])

    # Plot lines from initial neuron to target
    line_set = build_line_set_from_matches(pc0, pc1, matches)

    o3d.visualization.draw_geometries([line_set, pc0, pc1])

    return line_set


def visualize_tracks_two_matches(neurons0, neurons1, match0, match1, match2=None, offset=None):
    if offset is None:
        offset = np.array([0.001, 0, 0])
    import open3d as o3d
    n0, n1, pc0, pc1 = build_pair_of_point_clouds(neurons0, neurons1)

    # Plot lines from initial neuron to target
    lines0 = build_line_set_from_matches(pc0, pc1, match0)
    c0 = [[0, 1, 0] for _ in range(len(match0))]
    lines0.colors = o3d.utility.Vector3dVector(c0)

    pc0.translate(offset)
    pc1.translate(offset)
    lines1 = build_line_set_from_matches(pc0, pc1, match1)
    c1 = [[1, 0, 0] for _ in range(len(match1))]
    lines1.colors = o3d.utility.Vector3dVector(c1)
    pc0.translate(-offset)
    pc1.translate(offset)

    if match2 is not None:
        lines2 = build_line_set_from_matches(pc0, pc1, match2)
        c2 = [[0, 0, 1] for _ in range(len(match2))]
        lines2.colors = o3d.utility.Vector3dVector(c2)
        o3d.visualization.draw_geometries([lines0, lines1, lines2, pc0, pc1])
    else:
        o3d.visualization.draw_geometries([lines0, lines1, pc0, pc1])

    return lines0, lines1


def visualize_tracks_multiple_matches(all_pc, all_matches):
    """
    Visualizes tracks between multiple point clouds that have pair-wise matchings

    See also visualize_tracks_simple
    """
    import open3d as o3d

    all_lines = []
    for i, match in enumerate(all_matches):
        pc0 = all_pc[i]
        pc0.paint_uniform_color([0.5, 0.5, 0.5])
        pc1 = all_pc[i + 1]

        new_lines = build_line_set_from_matches(pc0, pc1, match)

        if new_lines.has_lines():
            all_lines.append(new_lines)

    pc1.paint_uniform_color([0, 0, 0])  # Last one

    pc_and_lines = copy.copy(all_pc)
    pc_and_lines.extend(all_lines)
    o3d.visualization.draw_geometries(pc_and_lines)


def build_line_set_from_matches(pc0, pc1, matches=None, color=None):
    import open3d as o3d
    if color is None:
        color = [0, 0, 1]
    try:
        # If point clouds are passed
        points = np.vstack((pc0.points, pc1.points))
        n0 = len(pc0.points)
    except AttributeError:
        # If numpy arrays are passed
        points = np.vstack((pc0, pc1))
        n0 = pc0.shape[0]

    # Convert matches to the coordinates of the combine point cloud
    if matches is None:
        matches = [[i, i] for i in range(n0)]
    # I've been having problems with overwriting the original list
    combined_matches = list([list(m) for m in matches])
    for i, match in enumerate(matches):
        combined_matches[i][1] = (n0 + match[1])

    # If the passed match data also has a weight, just take the first 2
    if len(combined_matches[0]) > 2:
        combined_matches = [c[:2] for c in combined_matches]
    colors = [color for i in range(len(matches))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(combined_matches),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visualize_cluster_labels(labels, pc):
    import open3d as o3d
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pc])


def visualize_clusters_from_dataframe(full_pc, clust_df, verbose=0, smallest_cluster=3, default_color=None):
    import open3d as o3d
    # Assign colors to the data frame based on cluster id
    if default_color is None:
        default_color = [0, 0, 0]
    max_label = clust_df['clust_ind'].max()
    clust_df['colors'] = list(plt.get_cmap("tab20")(pd.to_numeric(clust_df.clust_ind, downcast='float') / max_label))

    # Add colors to actual point cloud
    full_pc.paint_uniform_color(default_color)
    final_colors = np.asarray(full_pc.colors)

    for i, row in clust_df.iterrows():
        these_ind = row.all_ind_global
        if len(these_ind) < smallest_cluster:
            continue
        this_color = row['colors']
        if verbose >= 1:
            print(f"Color {this_color[:3]} for neurons {these_ind}")

        all_colors = np.vstack([this_color[:3] for i in these_ind])
        final_colors[these_ind, :] = all_colors

    full_pc.colors = o3d.utility.Vector3dVector(final_colors)

    o3d.visualization.draw_geometries([full_pc])

    return final_colors


def build_full_pc_from_list(all_keypoints_pcs):
    import open3d as o3d
    full_pc = o3d.geometry.PointCloud()
    for pc in all_keypoints_pcs:
        full_pc = full_pc + pc
    full_pc.paint_uniform_color([0.5, 0.5, 0.5])

    return full_pc


def draw_registration_result(source, target, transformation, base=None):
    import open3d as o3d
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([1, 0, 0])

    source_temp.transform(transformation)
    if base is not None:
        o3d.visualization.draw_geometries([base, source_temp, target_temp])
    else:
        o3d.visualization.draw_geometries([source_temp, target_temp])


##
## Using ReferenceFrame class
##

def plot_match_example(all_frames,
                       neuron_matches,
                       feature_matches,
                       which_frame_pair,
                       neuron0,
                       which_slice=15):
    """
    Shows 2d feature matches for an example neuron on one slice
    """
    # Get frame objects and data
    frame0 = all_frames[which_frame_pair[0]]
    frame1 = all_frames[which_frame_pair[1]]
    img0 = frame0.get_raw_data()[which_slice, ...]
    img1 = frame1.get_raw_data()[which_slice, ...]
    # Get the matching neuron in frame1, and the relevant indices
    this_match = np.array(neuron_matches[which_frame_pair])
    neuron1 = this_match[this_match[:, 0] == neuron0, 1]
    # Get keypoints and feature, then the subsets
    kp_ind0 = set(frame0.get_features_of_neuron(neuron0))
    kp0, kp1 = frame0.keypoints, frame1.keypoints

    these_features = []
    tmp = feature_matches[which_frame_pair]
    for this_fmatch in tmp:
        if this_fmatch.queryIdx in kp_ind0:
            these_features.append(this_fmatch)

    print(f"Displaying {len(these_features)}/{len(tmp)} matches")
    print(f"of a total of {len(frame0.features_to_neurons)} features")
    img3 = cv2.drawMatches(img0, kp0, img1, kp1, these_features, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(25, 45))
    plt.imshow(img3)
    plt.title(f"Feature matches for neuron {neuron0} to {neuron1} in frames {which_frame_pair}")
    plt.show()


def plot_matched_point_clouds(all_frames,
                              neuron_matches,
                              which_pair,
                              color=None,
                              actually_draw=True):
    """
    Plots the matched neurons between two frames
    """
    import open3d as o3d
    if color is None:
        color = [0, 0, 1]
    frame0 = all_frames[which_pair[0]]
    frame1 = all_frames[which_pair[1]]
    match = neuron_matches[which_pair]

    # Build point clouds and match lines
    pc0 = build_neuron_tree(frame0.neuron_locs, False)[1]
    pc0.paint_uniform_color([0.5, 0.5, 0.5])
    pc1 = build_neuron_tree(frame1.neuron_locs, False)[1]
    pc1.paint_uniform_color([0, 0, 0])
    match_lines = build_line_set_from_matches(pc0, pc1, matches=match, color=color)

    to_draw = [pc0, pc1, match_lines]
    if actually_draw:
        o3d.visualization.draw_geometries(to_draw)

    return to_draw


def plot_three_point_clouds(all_frames, neuron_matches, ind=(0, 1, 2)):
    """See also plot_matched_point_clouds"""
    import open3d as o3d
    options = {'all_frames': all_frames,
               'neuron_matches': neuron_matches,
               'actually_draw': False}
    if type(ind) == int:
        ind = (ind, ind + 1, ind + 2)

    k = (ind[0], ind[1])
    lines01 = plot_matched_point_clouds(which_pair=k, color=[1, 0, 0], **options)
    k = (ind[0], ind[2])
    lines02 = plot_matched_point_clouds(which_pair=k, color=[0, 1, 0], **options)
    k = (ind[1], ind[2])
    lines12 = plot_matched_point_clouds(which_pair=k, color=[0, 0, 1], **options)

    to_draw = list(lines01)
    to_draw.extend(lines02)
    to_draw.extend(lines12)
    o3d.visualization.draw_geometries(to_draw)

    return to_draw


##
## 2d visualizations
##

def match2quiver_from_frames(all_frames, all_matches, which_pair, actually_draw=True):
    """
    Plots neuron matches as a quiver plot using custom ReferenceFrame objects
    """

    n0_unmatched = all_frames[which_pair[0]].neuron_locs
    n1_unmatched = all_frames[which_pair[1]].neuron_locs
    matches = all_matches[which_pair]

    return match2quiver(matches, n0_unmatched, n1_unmatched, actually_draw)


def match2quiver(matches, n0_unmatched, n1_unmatched, actually_draw):
    """
    Plots neuron matches as a quiver plot
    """
    # Align the neuron locations via matches
    xyz = np.zeros((len(matches), 3), dtype=np.float32)  # Start point
    diff_vec = np.zeros((len(matches), 3), dtype=np.float32)  # Difference vector
    for m, match in enumerate(matches):
        try:
            v0 = n0_unmatched[int(match[0])]
            v1 = n1_unmatched[int(match[1])]
        except IndexError:
            v0 = n0_unmatched[match[0], :]
            v1 = n1_unmatched[match[1], :]
        xyz[m, :] = v0
        diff_vec[m, :] = v1 - v0
    # C = dat[:,2] / np.max(dat[:,1])
    if actually_draw:
        plt.quiver(xyz[:, 1], xyz[:, 2], diff_vec[:, 1], diff_vec[:, 2])
    # plt.title('Neuron matches based on features (has mistakes)')
    return xyz, diff_vec


##
## Histograms
##

def hist_of_tracklet_lens(df,
                          min_len=50,
                          bin_width=50,
                          num_frames=500,
                          ylim=100):
    """Histogram of the lengths of tracklets"""

    all_len = df['slice_ind'].apply(len)
    bins = int((num_frames - min_len) / bin_width)
    plt.hist(all_len[all_len > min_len], bins=bins)
    plt.ylim([0, ylim])
    plt.title(f"Lengths of individual tracks (minimum={min_len})")

    return all_len


def plot_tracklets_of_min_len(clust_df, min_len=20):
    all_num_tracks = defaultdict(int)

    for row in clust_df['slice_ind']:
        if len(row) < min_len:
            continue
        for ind in row:
            all_num_tracks[ind] += 1

    # all_num = np.zeros(num_frames)
    x = list(all_num_tracks.keys())
    y = list(all_num_tracks.values())
    # for k, val in all_num_tracks.items():
    #     all_num[k] = val

    plt.plot(x, y)
    plt.title(f"Number of tracks on each frame of min length {min_len}")
    plt.show()

    return x, y


def plot_tracklet_covering(clust_df, window_len=20):
    """
    Plots the number of tracklets that cover a frame AND a length afterwards
        i.e. frame n and n+window_len
    """
    all_num_tracks = defaultdict(int)
    for row in clust_df['slice_ind']:
        if len(row) < window_len:
            continue
        for ind in row:
            # Doesn't count the end of the track
            if ind == (row[-1] - window_len):
                break
            all_num_tracks[ind] += 1

    x = list(all_num_tracks.keys())
    y = list(all_num_tracks.values())

    plt.plot(x, y)
    plt.xlabel("Start of the window")
    plt.ylabel("Number of covering tracks")
    plt.title(f"Number of tracks covering a full window (length={window_len})")
    plt.show()

    return x, y


def plot_full_tracklet_covering(clust_df, window_len=20, num_frames=500):
    """
    Similar to plot_tracklet_covering but checks each frame individually
        Slower, but works for tracklets that may skip frames
    """
    x = list(range(num_frames - window_len))
    y = np.zeros_like(x)
    for i in x:
        which_frames = list(range(i, i + window_len + 1))

        def check_frames(vals):
            vals = set(vals)
            return all([f in vals for f in which_frames])

        tmp = clust_df['slice_ind'].apply(check_frames)
        y[i] = tmp.sum(axis=0)

    plt.plot(x, y)
    plt.xlabel("Start of the window")
    plt.ylabel("Number of covering tracks")
    plt.title(f"Number of tracks covering a full window (length={window_len})")
    plt.show()

    return x, y


def visualize_point_cloud_and_propagated_locations_from_frames(which_neuron, f0, f1, neuron0_trans):
    n0 = f0.neuron_locs
    n1 = f1.neuron_locs

    line, pc0_neuron, pc1_neuron, pc1_trans = visualize_point_cloud_and_propagated_locations(n0, n1, neuron0_trans,
                                                                                             which_neuron)

    return pc0_neuron, pc1_trans, pc1_neuron, line


def visualize_point_cloud_and_propagated_locations(n0, n1, neuron0_trans, which_neuron):
    import open3d as o3d
    # Original neurons
    pc0_neuron = o3d.geometry.PointCloud()
    pc0_neuron.points = o3d.utility.Vector3dVector(n0)
    pc0_neuron.paint_uniform_color([0.5, 0.5, 0.5])
    np.asarray(pc0_neuron.colors)[which_neuron, :] = [1, 0, 0]
    # Visualize the correspondence
    pc1_trans = o3d.geometry.PointCloud()
    pc1_trans.points = o3d.utility.Vector3dVector(neuron0_trans)
    pc1_trans.paint_uniform_color([0, 1, 0])
    corr = [(which_neuron, 0)]
    line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pc0_neuron, pc1_trans, corr)
    # Visualize this neuron with the new neurons as well
    pc1_neuron = o3d.geometry.PointCloud()
    pc1_neuron.points = o3d.utility.Vector3dVector(n1)
    pc1_neuron.paint_uniform_color([0, 0, 1])

    return line, pc0_neuron, pc1_neuron, pc1_trans


def visualize_tracklet_in_body(i_tracklet, i_frame,
                               tracklet_df, kp_df, all_frames,
                               to_plot=False,
                               to_error=True):
    if type(i_tracklet) != list:
        i_tracklet = [i_tracklet]

    for i_t in i_tracklet:
        tracklet_ind = get_indices_of_tracklet(i_t, tracklet_df)
        if not i_frame in tracklet_ind:
            print(f"{i_frame} is not in tracklet; try one of {tracklet_ind}")

    # Get this tracklet
    tracklet_xyz = []
    for i_t in i_tracklet:
        try:
            local_ind = tracklet_df['slice_ind'].iloc[i_t].index(i_frame)
            tracklet_xyz.append(tracklet_df['all_xyz'].iloc[i_t][local_ind])
        except IndexError:
            print(f"{i_frame} not in tracklet {i_t}")
            if to_error:
                raise ValueError

    # Get keypoints
    kp_xyz = []
    for i_kp in range(len(kp_df)):
        local_ind = kp_df['slice_ind'].iloc[i_kp].index(i_frame)
        kp_xyz.append(kp_df['all_xyz'].iloc[i_kp][local_ind])

    # Get all other neurons
    this_frame = all_frames[i_frame]
    _, pc_neurons, pc_tree = build_neuron_tree(this_frame.neuron_locs, False)
    pc_neurons.paint_uniform_color([0.5, 0.5, 0.5])

    # Color the tracklet and keypoint neurons
    for xyz in tracklet_xyz:
        [k, idx, _] = pc_tree.search_knn_vector_3d(xyz, 1)
        np.asarray(pc_neurons.colors)[idx[:], :] = [1, 0, 0]
    for xyz in kp_xyz:
        [k, idx, _] = pc_tree.search_knn_vector_3d(xyz, 1)
        np.asarray(pc_neurons.colors)[idx[:], :] = [0, 0, 1]

    if to_plot:
        import open3d as o3d
        o3d.visualization.draw_geometries([pc_neurons])

    return pc_neurons