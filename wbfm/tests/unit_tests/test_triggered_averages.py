import unittest

import numpy as np
import pandas as pd
import pytest

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, remove_short_state_changes
from wbfm.utils.traces.triggered_averages import TriggeredAverageIndices, \
    assign_id_based_on_closest_onset_in_split_lists


class TestBinaryVectors(unittest.TestCase):

    def setUp(self) -> None:
        long = 6
        short = 3
        # Create a series with short and long bouts
        beh = np.hstack([np.zeros(short), np.ones(long), np.zeros(long), np.ones(long), np.zeros(short), np.ones(short)])
        # Create a misannotated hole
        beh_complex = beh.copy()
        beh_complex[5] = 0
        beh_complex[11] = BehaviorCodes.UNKNOWN
        beh_complex[17] = BehaviorCodes.UNKNOWN

        self.beh = pd.Series(beh)
        self.beh_complex = pd.Series(beh_complex)
        self.num_frames = len(beh)

        self.opt = dict(trace_len=self.num_frames, ind_preceding=0, behavioral_state=1)

    def test_basic_contiguous_blocks(self):
        starts, ends = get_contiguous_blocks_from_column(self.beh, already_boolean=True)
        self.assertEqual([3, 15, 24], starts)
        self.assertEqual([9, 21, 27], ends)

    def test_basic_onsets(self):
        ind = TriggeredAverageIndices(self.beh, min_duration=0, **self.opt)

        onset_vec = list(ind.onset_vector())
        # beh    = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        expected = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.assertEqual(expected, onset_vec)

    def test_complex_contiguous_blocks(self):
        starts, ends = get_contiguous_blocks_from_column(self.beh_complex == 1, already_boolean=True)

        self.assertEqual(starts, [3, 6, 15, 18, 24])
        self.assertEqual(ends, [5, 9, 17, 21, 27])

    def test_complex_onsets(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=0, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [3, 6, 15, 24]
        self.assertEqual(expected, list(np.where(onset_vec)[0]))

    def test_complex_onsets_min_duration(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=3, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [6, 24]
        self.assertEqual(expected, list(np.where(onset_vec)[0]))

    def test_remove_small_gaps(self):
        beh = remove_short_state_changes(self.beh_complex == 1, min_length=2)
        expected = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        self.assertEqual(expected, list(beh))

    def test_complex_onsets_min_duration_fill_gaps(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=3, gap_size_to_remove=2, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [3, 15, 24]
        self.assertEqual(expected, list(np.where(onset_vec)[0]))

    def test_complex_onsets_short_only_fill_gaps(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=0, max_duration=4,
                                      gap_size_to_remove=2, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [24]
        self.assertEqual(expected, list(np.where(onset_vec)[0]))

    def test_complex_onsets_long_only_fill_gaps(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=4, max_duration=8,
                                      gap_size_to_remove=2, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [3, 15]
        self.assertEqual(expected, list(np.where(onset_vec)[0]))

    def test_assignment_based_on_onsets(self):
        short_onsets = np.array([7,  86, 121, 148])
        long_onsets = np.array([20, 169, 333, 492, 698, 984])
        rev_onsets = np.array([15, 66,  114,  130,  157,  288,  449,  654,  925, 1369])

        with pytest.raises(Exception):
            rev_id = assign_id_based_on_closest_onset_in_split_lists(short_onsets, long_onsets, rev_onsets)

            expected = {15: 0, 66: 1, 114: 0, 130: 0, 157: 0, 288: 1, 449: 1, 654: 1, 925: 1, 1369: 1}
            # expected = [0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
            self.assertEqual(expected, rev_id)
