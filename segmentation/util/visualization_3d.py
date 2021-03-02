import open3d as o3d
from DLC_for_WBFM.utils.feature_detection.visualization_tracks import visualize_clusters_from_dataframe
import numpy as np

def visualize_centroids(df, all_centroids):
    pc = o3d.geometry.PointCloud()

    for xyz in all_centroids:
        pc_tmp = o3d.geometry.PointCloud()
        pc_tmp.points = o3d.utility.Vector3dVector(np.array(xyz))
        pc = pc + pc_tmp

    pc.paint_uniform_color([0.5, 0.5, 0.5])
    visualize_clusters_from_dataframe(pc, df)
