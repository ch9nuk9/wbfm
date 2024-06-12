import unittest

import pandas as pd
import pytest

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.custom_errors import InvalidBehaviorAnnotationsError


class TestBehaviorCodes(unittest.TestCase):

    def setUp(self) -> None:
        # Generate a Series with valid BehaviorCode values
        self.beh_valid = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,
                                    BehaviorCodes.FWD | BehaviorCodes.DORSAL_TURN, BehaviorCodes.REV | BehaviorCodes.VENTRAL_TURN,
                                    BehaviorCodes.REV | BehaviorCodes.DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                                    BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                                    BehaviorCodes.UNKNOWN])

        # Generate a Series with invalid BehaviorCode values
        self.beh_invalid = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,
                                    BehaviorCodes.FWD | BehaviorCodes.DORSAL_TURN, BehaviorCodes.REV | BehaviorCodes.VENTRAL_TURN,
                                    BehaviorCodes.REV | BehaviorCodes.DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                                    BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                                    BehaviorCodes.UNKNOWN, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    def test_has_value(self):
        # Test that the valid behaviors have values
        BehaviorCodes.assert_all_are_valid(self.beh_valid)

    def test_not_all_valid(self):
        with pytest.raises(InvalidBehaviorAnnotationsError):
            BehaviorCodes.assert_all_are_valid(self.beh_invalid)

    def test_must_be_manually_annotated(self):
        # Test the specific behaviors that must be manually annotated
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD | BehaviorCodes.DORSAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV | BehaviorCodes.VENTRAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV | BehaviorCodes.DORSAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.SUPERCOIL))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.QUIESCENCE))

        # Test the behaviors that do not need to be manually annotated
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.NOT_ANNOTATED))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.UNKNOWN))

    def test_addition(self):
        # Make sure that the addition of two behaviors is correct
        self.assertEqual(BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN, BehaviorCodes.FWD + BehaviorCodes.VENTRAL_TURN)

        # Not_annotated and unknown should be ignored in addition
        self.assertEqual(BehaviorCodes.FWD, BehaviorCodes.FWD + BehaviorCodes.NOT_ANNOTATED)
        self.assertEqual(BehaviorCodes.FWD, BehaviorCodes.FWD + BehaviorCodes.UNKNOWN)

    def test_vector_addition(self):
        # Build two pandas series with behavior codes
        beh1 = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,
                            BehaviorCodes.FWD, BehaviorCodes.REV,
                            BehaviorCodes.REV, BehaviorCodes.SUPERCOIL,
                            BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                            BehaviorCodes.UNKNOWN])
        beh2 = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,
                            BehaviorCodes.DORSAL_TURN, BehaviorCodes.VENTRAL_TURN,
                            BehaviorCodes.DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                            BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                            BehaviorCodes.FWD])

        expected_beh = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,
                            BehaviorCodes.FWD | BehaviorCodes.DORSAL_TURN, BehaviorCodes.REV | BehaviorCodes.VENTRAL_TURN,
                            BehaviorCodes.REV | BehaviorCodes.DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                            BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                            BehaviorCodes.FWD])

        # Add the two series together
        beh_sum = beh1 + beh2

        # Make sure the addition is correct
        self.assertTrue(all([b_sum == b_exp for b_sum, b_exp in zip(beh_sum, expected_beh)]))

    def test_hashable(self):
        my_set = {BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN,}

        my_dict = {BehaviorCodes.FWD: 1, BehaviorCodes.REV: 2, BehaviorCodes.FWD | BehaviorCodes.VENTRAL_TURN: 3}
