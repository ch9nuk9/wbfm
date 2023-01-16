import unittest

import numpy as np

from wbfm.utils.projects.finished_project_data import ProjectData


class TestWormPostureClass(unittest.TestCase):

    def setUp(self):
        # Example project
        fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-23_spacer_7b_2per_agar/2022-11-23_worm8/project_config.yaml"
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

    def test_speeds(self):
        properties = [self.worm.worm_speed, self.worm.worm_angular_velocity]

        for p in properties:
            self.assertTrue(p(fluorescence_fps=False) is not None)
            self.assertTrue(p(fluorescence_fps=True) is not None)

    def test_speed_special_args(self):
        # print(self.worm.beh_annotation(fluorescence_fps=True) == 1)
        k0 = "signed"
        kwarg0 = [False, True]
        k1 = "subsample_before_derivative"
        kwarg1 = [False, True]
        k2 = "strong_smoothing"
        kwarg2 = [False, True]
        k3 = "signed"
        kwarg3 = [False, True]

        for v0 in kwarg0:
            for v1 in kwarg1:
                for v2 in kwarg2:
                    for v3 in kwarg3:
                        kwargs = {k0: v0, k1: v1, k2: v2, k3: v3}
                        self.assertTrue(self.worm.worm_speed(**kwargs) is not None)
