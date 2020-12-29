import open3d as o3d
from DLC_for_WBFM.utils.feature_detection.utils_features import build_neuron_tree
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd


def visualize_tracks(neurons0, neurons1, matches, to_plot_failed_lines=False):
    n0, pc_n0, tree_neurons0 = build_neuron_tree(neurons0)
    n1, pc_n1, tree_neurons1 = build_neuron_tree(neurons1)
    pc_n0.paint_uniform_color([0.5,0.5,0.5])
    pc_n1.paint_uniform_color([0,0,0])

    # Plot lines from initial neuron to target
    points = np.vstack((pc_n0.points,pc_n1.points))

    tmp = list(matches)
    for i,match in enumerate(matches):
        tmp[i][1] = (n0 + match[1])

    successful_lines = []
    failed_lines = []
    for row in tmp:
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
        o3d.visualization.draw_geometries([failed_line_set, successful_line_set, pc_n0, pc_n1])
    else:
        o3d.visualization.draw_geometries([successful_line_set, pc_n0, pc_n1])



def visualize_tracks_simple(pc0, pc1, matches):
    pc0.paint_uniform_color([1,0,0])
    pc1.paint_uniform_color([0,1,0])

    # Plot lines from initial neuron to target
    line_set = build_line_set_from_matches(pc0, pc1, matches)

    o3d.visualization.draw_geometries([line_set, pc0, pc1])


def visualize_tracks_multiple_matches(all_pc, all_matches):
    """
    Visualizes tracks between multiple point clouds that have pair-wise matchings

    See also visualize_tracks_simple
    """

    all_lines = []
    for i, match in enumerate(all_matches):
        pc0 = all_pc[i]
        pc0.paint_uniform_color([0.5,0.5,0.5])
        pc1 = all_pc[i+1]

        new_lines = build_line_set_from_matches(pc0, pc1, match)

        if new_lines.has_lines():
            all_lines.append(new_lines)

    pc1.paint_uniform_color([0,0,0]) # Last one


    pc_and_lines = copy.copy(all_pc)
    pc_and_lines.extend(all_lines)
    o3d.visualization.draw_geometries(pc_and_lines)


def build_line_set_from_matches(pc0, pc1, matches,
                                color=[0, 0, 1]):
    points = np.vstack((pc0.points,pc1.points))
    n0 = len(pc0.points)

    # Convert matches to the coordinates of the combine point cloud
    combined_matches = list(matches)
    for i,match in enumerate(matches):
        combined_matches[i][1] = (n0 + match[1])

    colors = [color for i in range(len(matches))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(combined_matches),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visualize_cluster_labels(labels, pc):

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pc])


def visualize_clusters_from_dataframe(full_pc, clust_df, verbose=0):
    # Assign colors to the data frame based on cluster id
    max_label = clust_df['clust_ind'].max()
    clust_df['colors'] = list(plt.get_cmap("tab20")(pd.to_numeric(clust_df.clust_ind, downcast='float') / max_label))

    # Add colors to actual point cloud
    full_pc.paint_uniform_color([0,0,0])
    final_colors = np.asarray(full_pc.colors)

    for i, row in clust_df.iterrows():
        these_ind = row.all_ind_global
        if len(these_ind) < 3:
            continue
        this_color = row['colors']
        if verbose >= 1:
            print(f"Color {this_color[:3]} for neurons {these_ind}")

        all_colors = np.vstack([this_color[:3] for i in these_ind])
        final_colors[these_ind,:] = all_colors

    full_pc.colors = o3d.utility.Vector3dVector(final_colors)

    o3d.visualization.draw_geometries([full_pc])


def draw_registration_result(source, target, transformation, base=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([1, 0, 0])

    source_temp.transform(transformation)
    if base is not None:
        o3d.visualization.draw_geometries([base, source_temp, target_temp])
    else:
        o3d.visualization.draw_geometries([source_temp, target_temp])
