import unittest

import pandas as pd

from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *


class TestTrackletStitching(unittest.TestCase):

    def setUp(self):
        print("Setting up...")

        dummy_opt = {'i0_xyz': [0, 0, 0],
                     'i1_xyz': [1, 0, 0],
                     'i1_prob': [0.9],
                     'this_point_cloud_offset': 2,
                     'next_point_cloud_offset': 5,
                     'i1_global': 10}

        next_ind, t0 = create_new_track(i0=0, i1=1,
                                        which_slice=0,
                                        next_clust_ind=1,
                                        **dummy_opt)
        next_ind, t1 = create_new_track(i0=2, i1=3,
                                        which_slice=2,
                                        next_clust_ind=next_ind,
                                        **dummy_opt)
        next_ind, t2 = create_new_track(i0=5, i1=6,
                                        which_slice=5,
                                        next_clust_ind=next_ind,
                                        **dummy_opt)

        self.t0, self.t1, self.t2 = t0, t1, t2

        self.df_test = t0.append(t1, ignore_index=True).append(t2, ignore_index=True)

    def test_single_match(self):
        matches = [[0, 1]]

        df_stitched = consolidate_tracklets(self.df_test, matches)

        self.assertEqual(len(df_stitched), 2)
        self.assertEqual(df_stitched['slice_ind'].loc[0], [0, 1, 2, 3])

        self.helper_check_total_length(df_stitched)

    def test_two_matches(self):
        matches = [[0, 1], [1, 2]]

        df_stitched = consolidate_tracklets(self.df_test, matches)

        self.assertEqual(len(df_stitched), 1)
        self.assertEqual(df_stitched['slice_ind'].loc[0], [0, 1, 2, 3, 5, 6])

        self.helper_check_total_length(df_stitched)

    def helper_check_total_length(self, new_df):
        len0 = sum(self.df_test['slice_ind'].apply(len))
        len1 = sum(new_df['slice_ind'].apply(len))

        self.assertEqual(len0, len1)
