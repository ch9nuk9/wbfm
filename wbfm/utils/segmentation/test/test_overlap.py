import numpy as np
import unittest
from wbfm.utils.segmentation.util.overlap import *

# when testing, one should assert summarizing results, such as number of outputs or lengths, instead of precise results

# some test need datasets. Since I code offline/locally, we will need to chagne the path to the datasets when porting!


class TestOverlaps(unittest.TestCase):
    """

    Tests for calculating the overlap between neurons predicted by algorithms.

    """

    # we want to test all overlap functions

    def test_calc_all_overlaps(self):
        # TODO: test 3d output array for unique numbers and zeroed out positions
        # could take real results and just use 3-4 slices
        pass

    def test_calc_best_overlap(self):

        example_mask_s0 = np.array([[1, 1], [1, 0]])
        example_masks_s1 = np.array([[2, 0], [2, 1]])

        best_overlap, best_mask = calc_best_overlap(example_mask_s0, example_masks_s1)

        expected_overlap = 2
        expected_mask = np.array([[1, 0], [1, 0]])

        self.assertEqual(expected_overlap, best_overlap)
        self.assertTrue((expected_mask == best_mask).all())

    def test_bipartite_stitching(self):
        pass

    def test_create_matches_list(self):
        slice1 = np.array([[1, 0, 3, 3],
                           [4, 4, 3, 3]])
        slice2 = np.array([[1, 2, 5, 5],
                           [17, 18, 5, 0]])

        test_output = create_matches_list(slice1, slice2)
        example_results = [[1, 1, 1], [4, 17, 1], [4, 18, 1], [3, 5, 3]]

        self.assertEqual(sorted(test_output), sorted(example_results))
        pass

    def test_renaming_stitched_array(self):
        pass

    def test_remove_short_neurons(self):

        pass

    def test_split_long_neurons(self):
        pass

    def test_calc_brightness(self):
        pass

    def test_calc_means_via_brightness(self):
        example_brightnesses = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        example_results = [4, 12]   # means of gaussian peaks
        
        test_results, _, _ = calc_means_via_brightnesses(example_brightnesses)
        self.assertEqual(example_results, test_results)

    def test_get_neuron_lengths_dict(self):
        # TODO also test an array with a neuron
        example_array = np.array([[[1, 0], [2, 5]],
                                  [[1, 0], [2, 3]],
                                  [[1, 2], [4, 5]]])

        example_result = {1: 3, 2: 3, 3: 1, 4: 1, 5: 2}
        test_result = get_neuron_lengths_dict(example_array)

        self.assertEqual(test_result, example_result)
        pass

    def test_remove_large_areas(self):
        plane = np.zeros((700, 900))
        plane[0:50, 0:50] = 1
        plane[100:250, 100:250] = 2
        plane[600:620, 450:470] = 3
        plane[120:150, 340:373] = 4
        plane[450:500, 600:621] = 5
        plane[420:430, 110:120] = 6
        plane[300:333, 600:603] = 7
        plane_areas = [np.count_nonzero(plane == u) for u in np.unique(plane) if u > 0]
        initial_areas = [2500, 22500, 400, 990, 1050, 100, 99]

        self.assertEqual(plane_areas, initial_areas)

        rm_plane_t1000 = remove_large_areas(plane, 1000)
        rm_areas_t1000 = [np.count_nonzero(rm_plane_t1000 == x) for x in np.unique(rm_plane_t1000) if x > 0]
        expected_areas_t1000 = [400, 990, 100, 99]
        self.assertEqual(rm_areas_t1000, expected_areas_t1000)

        rm_plane_t100 = remove_large_areas(plane, 100)
        rm_areas_t100 = [np.count_nonzero(rm_plane_t100 == y) for y in np.unique(rm_plane_t100) if y > 0]
        expected_areas_t100 = [99]
        self.assertEqual(rm_areas_t100, expected_areas_t100)

        # add a real-life example
        # array = np.load(stardist_prealigned)

    def test_array_dispatcher(self):
        pass

    def test_level2_overlap(self):
        pass

if __name__ == '__main__':
    unittest.main()