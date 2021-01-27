import numpy as np
import unittest
from overlap import calc_best_overlap, calc_all_overlaps, convert_to_3d

class TestOverlaps(unittest.TestCase):
    """

    Tests for calculating the overlap between neurons predicted by algorithms.

    """

    # we want to test all overlap functions

    # test for best overlap

    # test for min overlap

    # test calc_all_overlaps
    def test_calc_all_overlaps(self):
        # TODO: test 3d output array for unique numbers and zeroed out positions
        pass

    def test_calc_best_overlap(self):

        example_mask_s0 = np.array([[1, 1], [1, 0]])
        example_masks_s1 = np.array([[2, 0], [2, 1]])

        best_overlap, best_mask = calc_best_overlap(example_mask_s0, example_masks_s1)

        expected_overlap = 2
        expected_mask = np.array([[1, 0], [1, 0]])

        self.assertEqual(expected_overlap, best_overlap)
        self.assertTrue((expected_mask == best_mask).all())

    def test_convert_to_3d(self):

        # TODO: add output test for 3d; check for consecutive & unique numbers in array

        example_input = r'C:\Users\niklas.khoss\Desktop\stardist_testdata'
        expected_output_shape = (32, 700, 900)

        test_3d = convert_to_3d(example_input)

        self.assertEqual(test_3d.shape, expected_output_shape)

if __name__ == '__main__':
    unittest.main()