import unittest

import numpy as np

from wbfm.utils.segmentation.util.utils_postprocessing import split_neuron_and_update_dicts, calc_brightness, \
    get_neuron_lengths_dict


class TestSplitting(unittest.TestCase):

    def setUp(self) -> None:

        self.global_current_neuron = 2
        self.mask_array = np.array([[[1, 2], [0, 2]],
                                    [[1, 2], [2, 2]],
                                    [[1, 0], [0, 0]]])
        self.brightness_array = np.array([[[10, 10], [0, 10]],
                                         [[11, 11], [11, 11]],
                                         [[22, 22], [0, 0]]])

        # Calc brightness and so on
        self.neuron_brightnesses, self.neuron_z_planes, centroid_dict = calc_brightness(self.brightness_array,
                                                                                        self.mask_array)
        self.neuron_lengths = get_neuron_lengths_dict(self.mask_array)

    def test_brightnesses(self):
        neuron_brightnesses = {1: [10, 11, 22], 2: [10, 11]}
        self.assertEqual(neuron_brightnesses, self.neuron_brightnesses)
        neuron_lengths = {1: 3, 2: 2}
        self.assertEqual(neuron_lengths, self.neuron_lengths)
        neuron_z_planes = {1: [0, 1, 2], 2: [0, 1]}
        self.assertEqual(neuron_z_planes, self.neuron_z_planes)

    def test_simple_split(self):
        x_split_local_coord = 1
        neuron_id = 1
        new_neuron_lengths = {}

        global_current_neuron = split_neuron_and_update_dicts(self.global_current_neuron, self.mask_array,
                                                              self.neuron_brightnesses,
                                                              neuron_id, self.neuron_lengths,
                                                              self.neuron_z_planes, new_neuron_lengths,
                                                              x_split_local_coord)
        self.neuron_lengths.update(new_neuron_lengths)

        self.assertEqual(3, global_current_neuron)
        new_lengths = {1: 2, 2: 2, 3: 1}
        self.assertEqual(new_lengths, self.neuron_lengths)
        new_planes = {1: [0, 1], 2: [0, 1], 3: [2]}
        self.assertEqual(new_planes, self.neuron_z_planes)
        new_mask = np.array([[[1, 2], [0, 2]],
                            [[1, 2], [2, 2]],
                            [[3, 0], [0, 0]]])
        self.assertTrue((new_mask == self.mask_array).all())
