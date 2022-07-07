from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PointCloudPair:

    pts0: np.ndarray
    pts1: np.ndarray
    matches: np.ndarray = None

    confidence_threshold: float = 0.0

    @property
    def match_ordered_point_clouds(self):
        """Returns 2 numpy arrays of zxy point clouds, aligned as matched by final_matches"""
        pts0, pts1 = [], []
        n0, n1 = self.pts0, self.pts1
        for m in self.matches:
            if m[-1] < self.confidence_threshold:
                continue
            pts0.append(n0[m[0]])
            pts1.append(n1[m[1]])

        pts0, pts1 = np.array(pts0), np.array(pts1)
        return pts0, pts1

    @property
    def rigid_rotation_matrix_01(self):
        pts0, pts1 = self.match_ordered_point_clouds
        val, h, inliers = cv2.estimateAffine3D(pts0, pts1, confidence=0.999)
        return h

    @property
    def rigid_rotation_matrix_10(self):
        pts0, pts1 = self.match_ordered_point_clouds
        val, h, inliers = cv2.estimateAffine3D(pts1, pts0, confidence=0.999)
        return h

    @property
    def transformed_pts0(self):

        raw_cloud = self.pts0
        h = self.rigid_rotation_matrix_01

        transformed_pts0 = cv2.transform(np.array([raw_cloud]), h)[0]
        return transformed_pts0

    @property
    def transformed_pts1(self):
        raw_cloud = self.pts1
        h = self.rigid_rotation_matrix_10

        transformed_pts1 = cv2.transform(np.array([raw_cloud]), h)[0]
        return transformed_pts1

    def visualize_matches_open3d(self, rigidly_rotate=False):
        matches = self.matches
        if rigidly_rotate:
            pts0, pts1 = self.transformed_pts0, self.pts1
        else:
            pts0, pts1 = self.pts0, self.pts1
        from wbfm.utils.visualization.visualization_tracks import visualize_tracks
        visualize_tracks(pts0, pts1, matches=matches)


def visualize_two_matches(match0: PointCloudPair, match1: PointCloudPair):
    # Should probably just use: visualize_tracks_two_matches
    import open3d as o3d
    from wbfm.utils.visualization.visualization_tracks import visualize_tracks
    to_draw1 = visualize_tracks(match0.pts0, match0.pts1, matches=match0.matches, to_plot=False)
    to_draw2 = visualize_tracks(match1.pts0, match1.pts1, matches=match1.matches, to_plot=False)
    to_draw1.extend(to_draw2)
    o3d.visualization.draw_geometries(to_draw1)

