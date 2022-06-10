import unittest

from DLC_for_WBFM.utils.neuron_matching.matches_class import MatchesWithConfidence


class TestMatchesWithConfidence(unittest.TestCase):

    def setUp(self) -> None:
        i0 = [0, 1, 2, 3, 4]
        i1 = [5, 6, 7, 8, 9]
        i1_miss = [5, 6, 7, 8, -1]

        self.matches = MatchesWithConfidence.matches_from_array(list(zip(i0, i1)), 1)
        self.matches_missing = MatchesWithConfidence.matches_from_array(list(zip(i0, i1_miss)), 1)

    def test_init(self):
        pass

    def test_lengths(self):
        self.assertEqual(self.matches.get_num_matches(), 5)
        self.assertEqual(self.matches_missing.get_num_matches(), 4)
