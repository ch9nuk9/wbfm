import unittest

import numpy as np
import pandas as pd

from wbfm.utils.tracklets.high_performance_pandas import delete_tracklets_using_ground_truth, \
    dataframe_equal_including_nan


class TestOverwriteUsingGroundTruth(unittest.TestCase):

    def setUp(self) -> None:
        gt_dict = {('neuron_001', 'raw_neuron_ind_in_list'): 6*[0],
                   ('neuron_001', 'z'): 6*[0.1]}
        self.df_gt = pd.DataFrame(gt_dict)

        tracker_dict = {('tracklet_001', 'raw_neuron_ind_in_list'): [0, 0, 0, 1, 1, 1],
                        ('tracklet_001', 'z'): [0.1, 0.1, 0.1, 1.1, 1.1, 1.1],
                        ('tracklet_002', 'raw_neuron_ind_in_list'): [1, 1, 1, 0, 0, 0],
                        ('tracklet_002', 'z'): [0.1, 0.1, 0.1, 1.1, 1.1, 1.1]}
        self.df_tracker = pd.DataFrame(tracker_dict)

        target_dict = {('tracklet_001', 'raw_neuron_ind_in_list'): [np.nan, np.nan, np.nan, 1, 1, 1],
                        ('tracklet_001', 'z'): [np.nan, np.nan, np.nan, 1.1, 1.1, 1.1],
                        ('tracklet_002', 'raw_neuron_ind_in_list'): [1, 1, 1, np.nan, np.nan, np.nan],
                        ('tracklet_002', 'z'): [0.1, 0.1, 0.1, np.nan, np.nan, np.nan]}
        self.df_target = pd.DataFrame(target_dict)

    def test_delete(self):
        df_out, ind_to_delete, tracklet_name_to_array_index = \
            delete_tracklets_using_ground_truth(self.df_gt, self.df_tracker)

        self.assertTrue(dataframe_equal_including_nan(df_out, self.df_target).values.all())

        expected = [0, 1, 2]
        self.assertTrue(ind_to_delete['tracklet_001'] == expected)
        expected = [3, 4, 5]
        self.assertTrue(ind_to_delete['tracklet_002'] == expected)

        expected = [0, 1]
        self.assertTrue(tracklet_name_to_array_index['tracklet_001'] == expected)
        expected = [2, 3]
        self.assertTrue(tracklet_name_to_array_index['tracklet_002'] == expected)
