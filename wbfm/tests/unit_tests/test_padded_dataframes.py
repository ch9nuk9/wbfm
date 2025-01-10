import unittest

import numpy as np
import pandas as pd
from wbfm.utils.external.utils_pandas import to_sparse_multiindex, check_if_fully_sparse
from wbfm.utils.general.high_performance_pandas import insert_value_in_sparse_df, PaddedDataFrame, \
    split_single_sparse_tracklet, get_names_from_df


class TestPaddedDataFrame(unittest.TestCase):

    def setUp(self) -> None:
        self.neuron_name = 'neuron_001'
        self.cols = [(self.neuron_name, 'z'), (self.neuron_name, 'x')]
        tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                                (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.df = PaddedDataFrame.construct_from_basic_dataframe(tmp_df, name_mode='neuron',
                                                                 initial_empty_cols=1, default_num_to_add=10)
        self.df = to_sparse_multiindex(self.df)
        print(self.df.sparse)  # No error

    def test_type(self):
        self.assertEqual(type(self.df), PaddedDataFrame)

    def test_split(self):
        left, right = split_single_sparse_tracklet(10, self.df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))

        print(left.sparse)  # No error
        print(right.sparse)  # No error

    def test_double_split(self):
        left, right = split_single_sparse_tracklet(10, self.df)
        left2, right2 = split_single_sparse_tracklet(5, left)

        self.assertTrue(check_if_fully_sparse(left2))
        self.assertTrue(check_if_fully_sparse(right2))
        print(left2.sparse)  # No error
        print(right2.sparse)  # No error

    def test_multi_split(self):
        left = self.df.copy()
        for i in [10, 5, 2]:
            left, right = split_single_sparse_tracklet(i, left)
            self.assertTrue(check_if_fully_sparse(left))
        print(left.sparse)  # No error
        print(right.sparse)  # No error

        right = self.df.copy()
        for i in [10, 5, 2]:
            left, right = split_single_sparse_tracklet(i, right)
            self.assertTrue(check_if_fully_sparse(right))
        print(left.sparse)  # No error
        print(right.sparse)  # No error

    def test_add(self):
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.neuron_name, val=[10, 10])

        self.assertEqual(10, int(df[self.neuron_name, 'z'][0]))
        self.assertTrue(check_if_fully_sparse(df))
        print(df.sparse)

    def test_add_then_split(self):
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.neuron_name, val=10)

        left, right = split_single_sparse_tracklet(10, df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))

    def test_remove_then_split(self):
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.neuron_name, val=np.nan)

        left, right = split_single_sparse_tracklet(10, df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))

    def test_multiindex_as_column(self):
        cols = pd.MultiIndex.from_product([[self.neuron_name], ['x', 'z']])

        df = insert_value_in_sparse_df(self.df, index=0, columns=cols, val=np.nan)

        left, right = split_single_sparse_tracklet(10, df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))

    def test_insert_different_values(self):
        cols = pd.MultiIndex.from_product([[self.neuron_name], ['x', 'z']])

        df = insert_value_in_sparse_df(self.df, index=0, columns=cols, val=[1.5, 2.5])

        self.assertEqual(2.5, df[self.neuron_name, 'z'][0])
        self.assertEqual(1.5, df[self.neuron_name, 'x'][0])
        self.assertTrue(check_if_fully_sparse(df))
        print(df.sparse)

    def test_insert_different_values_opposite_order(self):
        cols = pd.MultiIndex.from_product([[self.neuron_name], ['z', 'x']])

        df = insert_value_in_sparse_df(self.df, index=0, columns=cols, val=[1.5, 2.5])

        self.assertEqual(1.5, df[self.neuron_name, 'z'][0])
        self.assertEqual(2.5, df[self.neuron_name, 'x'][0])
        self.assertTrue(check_if_fully_sparse(df))
        print(df.sparse)

    def test_default_empty_columns(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)
        self.assertEqual(self.df.remaining_empty_column_names[0], 'neuron_002')
        self.assertEqual(self.df.shape[1], 4)

    def test_add_empty_columns(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)
        self.df = self.df.copy_and_add_empty_columns(10)
        self.assertEqual(len(self.df.remaining_empty_column_names), 11)

    def test_add_empty_columns_after_checking(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)
        self.df = self.df.add_new_empty_column_if_none_left(min_empty_cols=10, num_to_add=10)
        self.assertEqual(len(self.df.remaining_empty_column_names), 11)

    def test_add_empty_columns_new_object(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)
        new_df = self.df.add_new_empty_column_if_none_left(min_empty_cols=10, num_to_add=10)
        self.assertEqual(len(self.df.remaining_empty_column_names), 11)
        self.assertEqual(len(new_df.remaining_empty_column_names), 11)

    def test_add_empty_columns_loop(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)
        new_df = self.df

        for i in range(10):
            new_df = new_df.add_new_empty_column_if_none_left(min_empty_cols=100, num_to_add=1)
            # Test inplace change
            self.assertEqual(len(self.df.remaining_empty_column_names), i + 2)
            # Test persistent change
            self.assertEqual(len(new_df.remaining_empty_column_names), i + 2)

        self.assertEqual(len(self.df.remaining_empty_column_names), 11)
        self.assertEqual(len(new_df.remaining_empty_column_names), 11)

    def test_inplace(self):
        self.assertEqual(len(self.df.remaining_empty_column_names), 1)

        tmp = self.df.copy_and_add_empty_columns(10)
        self.assertEqual(len(tmp.remaining_empty_column_names), 11)
        self.assertEqual(len(self.df.remaining_empty_column_names), 11)

        # Should do nothing
        tmp = self.df.add_new_empty_column_if_none_left(min_empty_cols=10, num_to_add=10)
        self.assertEqual(len(tmp.remaining_empty_column_names), 11)
        self.assertEqual(len(self.df.remaining_empty_column_names), 11)

        # Should add
        tmp = self.df.add_new_empty_column_if_none_left(min_empty_cols=100, num_to_add=10)
        self.assertEqual(len(tmp.remaining_empty_column_names), 21)
        self.assertEqual(len(self.df.remaining_empty_column_names), 21)

    def test_correctly_catch_not_enough_columns(self):
        # Split 1 is fine
        self.df.split_tracklet(5, self.neuron_name)

        # Split 2 requires making new columns
        self.df.split_tracklet(10, self.neuron_name)

    def test_multiple_splits(self):
        # Will require adding columns
        df_working_copy, all_new_names = self.df.split_tracklet_multiple_times([5, 10, 15], self.neuron_name)

        # Test added empty columns, in both objects
        initial_empty_cols = 1
        num_expected = df_working_copy.default_num_to_add - 3 + initial_empty_cols
        self.assertEqual(len(df_working_copy.remaining_empty_column_names), num_expected)
        self.assertEqual(len(self.df.remaining_empty_column_names), num_expected)

        # Test splits
        self.assertEqual(all_new_names[0], 'neuron_001')
        self.assertEqual(all_new_names[1], 'neuron_002')
        self.assertEqual(all_new_names[2], 'neuron_003')
        self.assertEqual(all_new_names[3], 'neuron_004')
        self.assertEqual(len(all_new_names), 4)

    def test_roundtrip_sparse(self):
        df_sparse = self.df.return_sparse_dataframe()

        current_names = get_names_from_df(df_sparse)
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

        df_new_pad = PaddedDataFrame.construct_from_basic_dataframe(df_sparse, name_mode='neuron',
                                                                    initial_empty_cols=1, default_num_to_add=10)
        current_names = get_names_from_df(df_new_pad)  # Returns the non-empty names
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

    def test_roundtrip_normal(self):
        df_normal = self.df.return_normal_dataframe()

        current_names = get_names_from_df(df_normal)
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

        df_new_pad = PaddedDataFrame.construct_from_basic_dataframe(df_normal, name_mode='neuron',
                                                                    initial_empty_cols=1, default_num_to_add=10)
        current_names = get_names_from_df(df_new_pad)  # Returns the non-empty names
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

    def test_sparse_saving(self):
        df_sparse = self.df.return_sparse_dataframe()
        df_sparse.to_pickle('df_pytest.pickle')

        df_loaded = pd.read_pickle('df_pytest.pickle')

        current_names = get_names_from_df(df_loaded)
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

        df_new_pad = PaddedDataFrame.construct_from_basic_dataframe(df_loaded, name_mode='neuron',
                                                                    initial_empty_cols=1, default_num_to_add=10)
        current_names = get_names_from_df(df_new_pad)  # Returns the non-empty names
        expected_names = [self.neuron_name]
        self.assertEqual(current_names, expected_names)

    def test_split_then_save(self):
        df_working_copy, all_new_names = self.df.split_tracklet_multiple_times([5, 10, 15], self.neuron_name)
        expected_names = ['neuron_001', 'neuron_002', 'neuron_003', 'neuron_004']

        current_names = get_names_from_df(df_working_copy)  # Returns the non-empty names
        self.assertEqual(current_names, expected_names)

        df_sparse = df_working_copy.return_sparse_dataframe()
        df_sparse.to_pickle('df_pytest.pickle')

        df_loaded = pd.read_pickle('df_pytest.pickle')

        current_names = get_names_from_df(df_loaded)
        self.assertEqual(current_names, expected_names)

        df_new_pad = PaddedDataFrame.construct_from_basic_dataframe(df_loaded, name_mode='neuron',
                                                                    initial_empty_cols=1, default_num_to_add=10)
        current_names = get_names_from_df(df_new_pad)  # Returns the non-empty names
        self.assertEqual(current_names, expected_names)
