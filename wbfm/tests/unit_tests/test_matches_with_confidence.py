import unittest

from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence


class TestMatchesWithConfidence(unittest.TestCase):

    def setUp(self) -> None:
        self.i0 = [0, 1, 2, 3, 4]
        self.i1 = [5, 6, 7, 8, 9]
        self.i1_miss = [5, 6, 7, 8, -1]
        self.conf = [0, 1, 1, 1, 0]

        self.matches = MatchesWithConfidence.matches_from_array(list(zip(self.i0, self.i1)), 1)
        self.matches_missing = MatchesWithConfidence.matches_from_array(list(zip(self.i0, self.i1_miss)), 1)

    def test_init(self):
        pass

    def test_lengths(self):
        self.assertEqual(self.matches.get_num_matches(), 5)
        self.assertEqual(self.matches_missing.get_num_matches(), 4)

    def test_conf(self):

        matches_with_conf = MatchesWithConfidence.matches_from_array(list(zip(self.i0, self.i1, self.conf)))
        self.assertEqual(len(matches_with_conf.confidence), len(matches_with_conf.indices0))
        self.assertEqual(len(matches_with_conf.confidence), 5)

    def test_conf_with_minimum(self):
        matches_with_conf = MatchesWithConfidence.matches_from_array(list(zip(self.i0, self.i1, self.conf)),
                                                                     minimum_confidence=0.5)
        self.assertEqual(len(matches_with_conf.confidence), len(matches_with_conf.indices0))
        self.assertEqual(len(matches_with_conf.confidence), 3)

    def test_conf_with_minimum_and_missing(self):
        matches_with_conf = MatchesWithConfidence.matches_from_array(list(zip(self.i0, self.i1_miss, self.conf)),
                                                                     minimum_confidence=0.5)
        self.assertEqual(len(matches_with_conf.confidence), len(matches_with_conf.indices0))
        self.assertEqual(len(matches_with_conf.confidence), 3)
