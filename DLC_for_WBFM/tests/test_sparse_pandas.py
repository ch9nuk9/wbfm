import unittest

import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.external.utils_pandas import to_sparse_multiindex, check_if_fully_sparse
from DLC_for_WBFM.utils.tracklets.high_performance_pandas import insert_value_in_sparse_df
from DLC_for_WBFM.utils.tracklets.utils_tracklets import split_single_sparse_tracklet


class TestSparsePandas(unittest.TestCase):

    def setUp(self) -> None:
        self.cols = ('neuron_000', 'z')
        self.df = pd.DataFrame({('neuron_000', 'z'): list(range(20))})
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
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.cols[0], val=10)

        self.assertEqual(10, int(df[self.cols][0]))
        self.assertTrue(check_if_fully_sparse(df))
        print(df.sparse)

    def test_add_then_split(self):
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.cols[0], val=10)

        left, right = split_single_sparse_tracklet(10, df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))

    def test_remove_then_split(self):
        df = insert_value_in_sparse_df(self.df, index=0, columns=self.cols[0], val=np.nan)

        left, right = split_single_sparse_tracklet(10, df)
        self.assertTrue(check_if_fully_sparse(left))
        self.assertTrue(check_if_fully_sparse(right))
        