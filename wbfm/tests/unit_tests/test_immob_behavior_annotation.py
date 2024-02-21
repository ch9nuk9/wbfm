import unittest

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.projects.finished_project_data import ProjectData


class TestImmobBehavior(unittest.TestCase):

    def setUp(self) -> None:
        fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13"
        self.project_data = ProjectData.load_final_project_data_from_config(fname)

    def test_immob_behavior_annotation(self):
        # Check they are not empty
        self.assertTrue(self.project_data.worm_posture_class.has_beh_annotation)

    def test_only_low_res(self):
        # Immobilized behavior is calculated from the traces, thus should be low-res only
        self.assertTrue(self.project_data.worm_posture_class.beh_annotation_already_converted_to_fluorescence_fps)

    def test_has_rev_and_fwd(self):
        # This is the basic set of annotations it should have
        beh_vector = self.project_data.worm_posture_class.beh_annotation(fluorescence_fps=True)
        self.assertTrue(BehaviorCodes.REV in beh_vector)
        self.assertTrue(BehaviorCodes.FWD in beh_vector)
