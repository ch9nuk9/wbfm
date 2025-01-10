import unittest

import numpy as np
import pandas as pd
from wbfm.utils.external.utils_pandas import get_times_of_conflicting_dataframes
from wbfm.utils.general.high_performance_pandas import insert_value_in_sparse_df, get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once


class TestPaddedDataFrame(unittest.TestCase):

    def setUp(self) -> None:
        length = 20

        self.tracklet_names = ['tracklet_0000001', 'tracklet_0000002', 'tracklet_0000003']
        tmp_dict = {}
        for name in self.tracklet_names:
            update = {(name, 'z'): list(range(length)), (name, 'x'): 10*np.array(list(range(length)))}
            tmp_dict.update(update)
        df = pd.DataFrame(tmp_dict).astype(pd.SparseDtype(float, np.nan))

        # Add gaps to test splitting and conflict seeking
        df = insert_value_in_sparse_df(df, index=list(range(5)), columns='tracklet_0000002', val=np.nan)
        df = insert_value_in_sparse_df(df, index=list(range(15, length)), columns='tracklet_0000003', val=np.nan)

        self.df = df
        self.tracklet_list = [self.df[[n]] for n in self.tracklet_names]

    def test_init(self):
        pass

    def test_get_times_of_conflicting_dataframes(self):
        overlapping_tracklet_conflict_points = get_times_of_conflicting_dataframes(self.tracklet_list,
                                                                                   self.tracklet_names)

        expected = dict(tracklet_0000001=[5, 15],
                        tracklet_0000002=[15],
                        tracklet_0000003=[5])
        self.assertEqual(overlapping_tracklet_conflict_points, expected)

        df_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(self.df,
                                                                                overlapping_tracklet_conflict_points)

        self.assertEqual(len(all_new_tracklets), 7)
        self.assertEqual(df_split.shape[0], self.df.shape[0])

        # Should be no further conflicts
        new_tracklet_names = get_names_from_df(df_split)
        overlapping_tracklet_conflict_points = get_times_of_conflicting_dataframes(all_new_tracklets,
                                                                                   new_tracklet_names)
        self.assertEqual(overlapping_tracklet_conflict_points, dict())

        # Check if the new tracklets are correct
        expected_tracklet_names = ['tracklet_0000004', 'tracklet_0000005', 'tracklet_0000006', 'tracklet_0000007',
                                   'tracklet_0000008', 'tracklet_0000009', 'tracklet_0000010']
        self.assertEqual(new_tracklet_names, expected_tracklet_names)

        # Check if the values are correct
        print(self.df)
        print(df_split)


