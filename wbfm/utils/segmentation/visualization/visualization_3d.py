import open3d as o3d
from wbfm.utils.visualization.visualization_tracks import visualize_clusters_from_dataframe
import numpy as np
import wbfm.utils.segmentation.util.overlap as ol


def visualize_centroids(df, all_centroids):
    pc = o3d.geometry.PointCloud()

    for xyz in all_centroids:
        if xyz:
            pc_tmp = o3d.geometry.PointCloud()
            pc_tmp.points = o3d.utility.Vector3dVector(np.array(xyz))
            pc = pc + pc_tmp

    pc.paint_uniform_color([0.5, 0.5, 0.5])
    visualize_clusters_from_dataframe(pc, df)


# sd_3d = np.load('C:\\Segmentation_working_area\\data\\stardist_raw\\3d\\preprocessed_versatile_fluo_3d.npy')
# sd_rm = ol.remove_large_areas(sd_3d)
# sd_stitched = ol.bipartite_stitching(sd_rm)
