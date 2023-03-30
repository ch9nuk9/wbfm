import unittest

import numpy as np
import pandas as pd
from wbfm.utils.external.utils_pandas import check_if_fully_sparse, to_sparse_multiindex
from wbfm.utils.projects.utils_consolidation import calc_split_dict_z_threshold
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once, get_time_overlap_of_candidate_tracklet


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

    # def test_pipeline_split_function(self):
    #     # Use wiggle_tracklet_endpoint_to_remove_conflict to split the tracklet
    #     # and check that the split is correct
    #
    #     # First, add some nan points to the dataframe
    #     df = self.df.copy()
    #     df.loc[:5, (self.neuron_name, slice(None))] = np.nan
    #     covering_tracklet_names = [self.neuron_name]
    #
    #     # Second, add the candidate tracklet that will be added to this neuron
    #     # Should be a multi-index dataframe with columns 'x' and 'z', of the same length as df
    #     # The index should be the same as df
    #     candidate_name = 'tracklet_0000001'
    #     df_candidate = pd.DataFrame({(candidate_name, 'z'): list(range(20)),
    #                                  (candidate_name, 'x'): 10*np.array(list(range(20)))})
    #     # Create an overlap from 6 to 7
    #     true_split_point = 5
    #     candidate_end_point = 9
    #     df_candidate.loc[candidate_end_point:, (candidate_name, slice(None))] = np.nan
    #     # Concatenate the two dataframes
    #     df_tracklets = pd.concat([df, df_candidate], axis=1)
    #
    #     i_tracklet = get_names_from_df(df_tracklets).index(candidate_name)
    #
    #     # Go through the steps of the pipeline. See: calc_covering_from_distances
    #     t = df_tracklets.index
    #     is_nan = df_tracklets[candidate_name]['x'].isnull()
    #     newly_covered_times = list(t[~is_nan])
    #
    #     # Make sure there is a conflict
    #     time_conflicts = get_time_overlap_of_candidate_tracklet(
    #         candidate_name, covering_tracklet_names, df_tracklets
    #     )
    #     self.assertTrue(len(time_conflicts) > 0)
    #
    #     # Split the tracklet
    #     allowed_number_of_conflict_points = 3
    #
    #     new_candidate_name, df_tracklets_new, i_tracklet_new, successfully_split = \
    #         wiggle_tracklet_endpoint_to_remove_conflict(allowed_number_of_conflict_points, candidate_name,
    #                                                     time_conflicts, df_tracklets, i_tracklet, newly_covered_times)
    #
    #     # Check that the split was successful
    #     self.assertTrue(successfully_split)
    #
    #     # Check that the candidate name was kept the same, because they split was on the past (left) side
    #     self.assertEqual(candidate_name, new_candidate_name)
    #
    #     # Check that the index is unchanged
    #     self.assertEqual(i_tracklet, i_tracklet_new)
    #
    #     # Check that the dataframe has a new column for the split tracklet
    #     new_names = get_names_from_df(df_tracklets_new)
    #     expected_names = [self.neuron_name, candidate_name, 'tracklet_0000002']
    #     self.assertEquals(expected_names, new_names)
    #
    #     print(df_tracklets_new)
    #
    #     # Check that the split point is correct by checking the non-nan values in each tracklet
    #     expected_nonnan_new_candidate = list(range(true_split_point))
    #     nonnan_new_candidate = list(df_tracklets_new[candidate_name].dropna(axis=0).index)
    #     self.assertEqual(expected_nonnan_new_candidate, nonnan_new_candidate)
    #
    #     expected_nonnan_new_tracklet = list(range(true_split_point, candidate_end_point+1))
    #     nonnan_new_tracklet = list(df_tracklets_new['tracklet_0000002'].dropna(axis=0).index)
    #     self.assertEqual(expected_nonnan_new_tracklet, nonnan_new_tracklet)
