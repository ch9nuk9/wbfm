import unittest

import numpy as np
import pandas as pd
from wbfm.utils.external.utils_pandas import to_sparse_multiindex, check_if_fully_sparse
from wbfm.utils.general.high_performance_pandas import insert_value_in_sparse_df, split_single_sparse_tracklet


class TestSparsePandas(unittest.TestCase):

    def setUp(self) -> None:
        self.neuron_name = 'neuron_001'
        self.cols = [(self.neuron_name, 'z'), (self.neuron_name, 'x')]
        self.df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                                (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.assertRaises(AttributeError, lambda: self.df.sparse)
        self.df = to_sparse_multiindex(self.df)
        print(self.df.sparse)  # No error

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
