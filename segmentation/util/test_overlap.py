import numpy as np
import unittest
from overlap import calc_best_overlap, calc_all_overlaps

class TestOverlaps(unittest.TestCase):

    # we want to test all overlap functions

    # test for best overlap

    # test for min overlap

    # test calc_all_overlaps
    def test_calc_all_overlaps(self):
        example_data = 0

    def test_calc_best_overlap(self):
        example_mask_s0 = np.array([[1, 1], [1, 0]])
        example_masks_s1 = np.array([[2, 0], [2, 1]])

        best_ind, best_overlap, best_mask = calc_best_overlap(example_mask_s0, example_masks_s1)

        expected_ind = 2
        expected_overlap = 2
        expected_mask = np.array([[1, 0], [1, 0]])

        self.assertEqual(expected_ind, best_ind)
        self.assertEqual(expected_overlap, best_overlap)
        self.assertTrue((expected_mask == best_mask).all())

if __name__ == '__main__':
    unittest.main()