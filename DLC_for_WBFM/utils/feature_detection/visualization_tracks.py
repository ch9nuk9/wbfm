import open3d as o3d
from DLC_for_WBFM.utils.feature_detection.utils_features import build_neuron_tree
import numpy as np
import matplotlib.pyplot as plt


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


def visualize_cluster_labels(labels, pc):

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pc])
