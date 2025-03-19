"""
Testing area for all 'utils_accuracy.py' related functions.

"""
import unittest
import numpy as np
from wbfm.utils.segmentation.util.utils_accuracy import *


class TestAccuracy(unittest.TestCase):

    def test_seg_accuracy(self):
        pass

    def test_create_3d_match(self):
        dataset1 = np.array([[[1, 1, 3],
                              [1, 1, 0],
                              [0, 2, 2]],
                             [[1, 1, 3],
                              [1, 2, 0],
                              [0, 2, 2]],
                             [[1, 1, 3],
                              [1, 1, 2],
                              [0, 2, 2]]])

        dataset2 = np.array([[[4, 4, 0], [0, 4, 4], [5, 5, 0]],
                             [[4, 4, 0], [4, 4, 5], [6, 5, 5]],
                             [[4, 4, 0], [4, 5, 5], [6, 5, 5]]])

        d1_matches, d1_volumes = create_3d_match_dict(dataset1, dataset2)
        example_matches = {1: [4, 5], 2: [4, 5], 3: []}
        example_volumes = {1: [9, 1], 2: [1, 6], 3: []}

        d2_matches, d2_volumes = create_3d_match_dict(dataset2, dataset1)
        example_matches_2 = {4: [1, 2], 5: [1, 2], 6: []}
        example_volumes_2 = {4: [9, 1], 5: [1, 6], 6: []}

        self.assertEqual(example_matches_2, d2_matches)
        self.assertEqual(example_volumes_2, d2_volumes)

        self.assertEqual(d1_matches, example_matches)
        self.assertEqual(d1_volumes, example_volumes)
