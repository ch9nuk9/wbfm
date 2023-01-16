import unittest

import numpy as np

from wbfm.utils.projects.finished_project_data import ProjectData


class TestWormPostureClass(unittest.TestCase):

    def setUp(self):
        # Example project
        fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
        project_data = ProjectData.load_final_project_data_from_config(fname)
        self.project_data = project_data
        self.worm = project_data.worm_posture_class

    def test_files_found(self):
        self.assertTrue(self.worm.has_beh_annotation)
        self.assertTrue(self.worm.has_full_kymograph)

    def test_files_loaded(self):
        properties = [self.worm.centerlineX, self.worm.centerlineY, self.worm.curvature, self.worm.stage_position,
                      self.worm.beh_annotation, self.worm.centerline_absolute_coordinates]

        for p in properties:
            self.assertTrue(p() is not None)

    def test_subsampling(self):
        properties = [self.worm.centerlineX, self.worm.centerlineY, self.worm.curvature, self.worm.stage_position,
                      self.worm.beh_annotation]

        for p in properties:
            self.assertTrue(p(fluorescence_fps=True) is not None)
