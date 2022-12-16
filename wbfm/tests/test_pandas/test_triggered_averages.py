import unittest

import numpy as np
import pandas as pd

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, remove_short_state_changes
from wbfm.utils.traces.triggered_averages import TriggeredAverageIndices


class TestBinaryVectors(unittest.TestCase):

    def setUp(self) -> None:
        long = 6
        short = 3
        # Create a series with short and long bouts
        beh = np.hstack([np.zeros(short), np.ones(long), np.zeros(long), np.ones(long), np.zeros(short), np.ones(short)])
        # Create a misannotated hole
        beh_complex = beh.copy()
        beh_complex[5] = 0
        beh_complex[11] = -1
        beh_complex[17] = -1

        self.beh = pd.Series(beh)
        self.beh_complex = pd.Series(beh_complex)
        self.num_frames = len(beh)

        self.opt = dict(trace_len=self.num_frames, ind_preceding=0, behavioral_state=1, DEBUG=True)

    def test_basic_contiguous_blocks(self):
        starts, ends = get_contiguous_blocks_from_column(self.beh, already_boolean=True)
        self.assertEqual(starts, [3, 15, 24])
        self.assertEqual(ends, [9, 21, 27])

    def test_basic_onsets(self):
        ind = TriggeredAverageIndices(self.beh, min_duration=0, **self.opt)

        onset_vec = list(ind.onset_vector())
        # beh    = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        expected = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.assertEqual(onset_vec, expected)

    def test_complex_contiguous_blocks(self):
        starts, ends = get_contiguous_blocks_from_column(self.beh_complex == 1, already_boolean=True)
        # print()
        # print([int(i) for i in self.beh_complex])
        self.assertEqual(starts, [3, 6, 15, 18, 24])
        self.assertEqual(ends, [5, 9, 17, 21, 27])

    def test_complex_onsets(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=0, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [3, 6, 15, 24]
        self.assertEqual(list(np.where(onset_vec)[0]), expected)

    def test_complex_onsets_min_duration(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=3, **self.opt)
        onset_vec = list(ind.onset_vector())
        expected = [6, 24]
        self.assertEqual(list(np.where(onset_vec)[0]), expected)

    def test_remove_small_gaps(self):
        print()
        beh = remove_short_state_changes(self.beh_complex == 1, min_length=2, DEBUG=True)
        print()
        print([int(i) for i in self.beh_complex])
        print([int(i) for i in beh])

    def test_complex_onsets_min_duration_fill_gaps(self):
        ind = TriggeredAverageIndices(self.beh_complex, min_duration=3, gap_size_to_remove=1, **self.opt)
        onset_vec = list(ind.onset_vector())
        print()
        print([int(i) for i in self.beh_complex])
        print([int(i) for i in ind.binary_state])
        print([int(i) for i in onset_vec])
        print(np.where(onset_vec))
        expected = [6, 24]
        self.assertEqual(list(np.where(onset_vec)[0]), expected)