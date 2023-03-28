import unittest

import numpy as np
import pandas as pd
from wbfm.utils.external.utils_pandas import check_if_fully_sparse, to_sparse_multiindex
from wbfm.utils.projects.utils_consolidation import calc_split_dict_z_threshold
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once


class TestTrackletSplitting(unittest.TestCase):

    def setUp(self) -> None:
        self.neuron_name = 'neuron_001'
        self.cols = [(self.neuron_name, 'z'), (self.neuron_name, 'x')]
        tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                              (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.df = tmp_df

    def test_split_several_at_once(self):
        split_list_dict = {self.neuron_name: [10, 15]}
        df_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(self.df, split_list_dict, name_mode='neuron')

        # Generated names are correct
        expected_names = ['neuron_002', 'neuron_003', 'neuron_004']
        split_names = get_names_from_df(df_split)
        self.assertEqual(expected_names, split_names)

        # Two columns per neuron; test that the lengths are correct
        expected_lens = [10, 10, 5, 5, 5, 5]
        split_lens = list(df_split.count())
        self.assertEqual(expected_lens, split_lens)

        # Specific indices are correct
        expected_indices = [list(range(10)), list(range(10, 15)), list(range(15, 20))]
        split_indices = [list(t.dropna(axis=0).index) for t in all_new_tracklets]
        self.assertEqual(expected_indices, split_indices)

        # Test that they are sparse
        df_split_sparse = to_sparse_multiindex(df_split)
        self.assertTrue(check_if_fully_sparse(df_split_sparse))

    def test_split_out_of_order(self):
        split_list_dict = {self.neuron_name: [15, 10]}
        df_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(self.df, split_list_dict, name_mode='neuron')

        expected_names = ['neuron_002', 'neuron_003', 'neuron_004']
        split_names = get_names_from_df(df_split)
        self.assertEqual(expected_names, split_names)

        # Two columns per neuron
        expected_lens = [10, 10, 5, 5, 5, 5]
        split_lens = list(df_split.count())
        self.assertEqual(expected_lens, split_lens)

        # Specific indices
        expected_indices = [list(range(10)), list(range(10, 15)), list(range(15, 20))]
        split_indices = [list(t.dropna(axis=0).index) for t in all_new_tracklets]
        self.assertEqual(expected_indices, split_indices)

        # Test that they are sparse
        df_split_sparse = to_sparse_multiindex(df_split)
        self.assertTrue(check_if_fully_sparse(df_split_sparse))

    def test_split_based_on_z(self):
        # Modify the z column of the dataframe to create a large jump
        df = self.df.copy()
        df[(self.neuron_name, 'z')] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        split_list_dict = calc_split_dict_z_threshold(df, self.neuron_name, z_threshold=3)

        expected_split_dict = {self.neuron_name: [10, 15]}
        self.assertEqual(expected_split_dict, split_list_dict)

        # Split and check all the same as above
        df_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(df, split_list_dict, name_mode='neuron')

        # Generated names are correct
        expected_names = ['neuron_002', 'neuron_003', 'neuron_004']
        split_names = get_names_from_df(df_split)
        self.assertEqual(expected_names, split_names)

        # Two columns per neuron
        expected_lens = [10, 10, 5, 5, 5, 5]
        split_lens = list(df_split.count())
        self.assertEqual(expected_lens, split_lens)

        # Specific indices
        expected_indices = [list(range(10)), list(range(10, 15)), list(range(15, 20))]
        split_indices = [list(t.dropna(axis=0).index) for t in all_new_tracklets]
        self.assertEqual(expected_indices, split_indices)

        # Check that the z values are correct
        expected_z_values = [list(range(10)), list(range(5)), list(range(5))]
        split_z_values = [list(t.dropna(axis=0)['z']) for t in all_new_tracklets]
        self.assertEqual(expected_z_values, split_z_values)

        # Test that they are sparse
        df_split_sparse = to_sparse_multiindex(df_split)
        self.assertTrue(check_if_fully_sparse(df_split_sparse))
