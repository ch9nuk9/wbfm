import unittest

from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence, get_mismatches


class TestTrackingMismatches(unittest.TestCase):

    def setUp(self) -> None:
        self.gt = MatchesWithConfidence.matches_from_array([[0, 0], [1, 1], [2, 2]], 1)
        self.model_missing = MatchesWithConfidence.matches_from_array([[0, 0], [1, 1]], 1)
        self.model_extra = MatchesWithConfidence.matches_from_array([[0, 0], [1, 1], [2, 2], [3, 3]], 1)
        self.model_wrong = MatchesWithConfidence.matches_from_array([[0, 0], [1, 1], [2, 3]], 1)

    def test_class_init(self):
        pass

    def test_missing(self):
        correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = \
            get_mismatches(self.gt, self.model_missing)

        self.assertEqual(gt_matches_different_model, [])
        self.assertEqual(model_matches_different_gt, [])
        self.assertEqual(model_matches_no_gt, [])
        self.assertEqual(gt_matches_no_model, [[2, 2]])

    def test_extra(self):
        correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = \
            get_mismatches(self.gt, self.model_extra)

        self.assertEqual(gt_matches_different_model, [])
        self.assertEqual(model_matches_different_gt, [])
        self.assertEqual(model_matches_no_gt, [[3, 3]])
        self.assertEqual(gt_matches_no_model, [])

    def test_wrong(self):
        correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = \
            get_mismatches(self.gt, self.model_wrong)

        self.assertEqual(gt_matches_different_model, [[2, 2]])
        self.assertEqual(model_matches_different_gt, [[2, 3]])
        self.assertEqual(model_matches_no_gt, [])
        self.assertEqual(gt_matches_no_model, [])
