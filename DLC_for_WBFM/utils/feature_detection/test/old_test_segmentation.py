import unittest

import cv2
import numpy as np
import tifffile
import os
import copy
import open3d as o3d
import pandas as pd

from DLC_for_WBFM.utils.feature_detection.utils_features import build_feature_tree
from DLC_for_WBFM.utils.feature_detection.utils_detection import build_point_clouds_for_volume, build_correspondence_icp


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        print("Setting up...")

        fname = 'img100.tif'
        self.num_slices = 33
        self.alpha = 0.15
        # self.fname1 = 'img101.tif'

        with tifffile.TiffFile(fname) as tif:
            self.dat = tif.asarray()

        all_keypoints_pcs = build_point_clouds_for_volume(self.dat,
                                                          self.num_slices,
                                                          self.alpha)
        self.all_keypoints_pcs = all_keypoints_pcs
        print("Finished setting up.")


    def test_keypoints(self):
        # Build point clouds for each plane
        all_keypoints_pcs = self.all_keypoints_pcs

        self.assertTrue(len(all_keypoints_pcs)==self.num_slices)
        self.assertTrue(type(all_keypoints_pcs[0]) == o3d.cpu.pybind.geometry.PointCloud)

        self.assertEqual(np.asarray(all_keypoints_pcs[0].points).shape, (67,3))
        self.assertEqual(np.asarray(all_keypoints_pcs[1].points).shape, (0,3))
        self.assertEqual(np.asarray(all_keypoints_pcs[2].points).shape, (1,3))
        self.assertEqual(np.asarray(all_keypoints_pcs[-1].points).shape, (22,3))


    def test_icp(self):

        all_keypoints_pcs = self.all_keypoints_pcs
        all_icp = build_correspondence_icp(all_keypoints_pcs)

        self.assertTrue(len(all_icp) == self.num_slices-1)
        self.assertTrue(type(all_icp[0]) == o3d.cpu.pybind.pipelines.registration.RegistrationResult)

        self.assertEqual(np.asarray(all_icp[0].correspondence_set).shape, (0,2))
        self.assertEqual(np.asarray(all_icp[5].correspondence_set).shape, (1,2))

        self.assertTrue(np.allclose(np.asarray(all_icp[5].correspondence_set), [[0,0]]))
