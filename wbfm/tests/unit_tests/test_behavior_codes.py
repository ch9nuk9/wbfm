import unittest

import pandas as pd
import pytest

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.custom_errors import InvalidBehaviorAnnotationsError


class TestBehaviorCodes(unittest.TestCase):

    def setUp(self) -> None:
        # Generate a Series with valid BehaviorCode values
        self.beh_valid = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD_VENTRAL_TURN,
                                    BehaviorCodes.FWD_DORSAL_TURN, BehaviorCodes.REV_VENTRAL_TURN,
                                    BehaviorCodes.REV_DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                                    BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                                    BehaviorCodes.UNKNOWN])

        # Generate a Series with invalid BehaviorCode values
        self.beh_invalid = pd.Series([BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.FWD_VENTRAL_TURN,
                                    BehaviorCodes.FWD_DORSAL_TURN, BehaviorCodes.REV_VENTRAL_TURN,
                                    BehaviorCodes.REV_DORSAL_TURN, BehaviorCodes.SUPERCOIL,
                                    BehaviorCodes.QUIESCENCE, BehaviorCodes.NOT_ANNOTATED,
                                    BehaviorCodes.UNKNOWN, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    def test_has_value(self):
        for beh in self.beh_valid:
            self.assertTrue(BehaviorCodes.has_value(beh))

    def test_not_all_valid(self):
        with pytest.raises(InvalidBehaviorAnnotationsError):
            BehaviorCodes.assert_all_are_valid(self.beh_invalid)

    def test_must_be_manually_annotated(self):
        # Test the specific behaviors that must be manually annotated
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD_VENTRAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD_DORSAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV_VENTRAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV_DORSAL_TURN))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.SUPERCOIL))
        self.assertTrue(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.QUIESCENCE))

        # Test the behaviors that do not need to be manually annotated
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.FWD))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.REV))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.NOT_ANNOTATED))
        self.assertFalse(BehaviorCodes.must_be_manually_annotated(BehaviorCodes.UNKNOWN))




