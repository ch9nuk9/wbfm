import unittest
import numpy as np
from parameterized import parameterized_class

from wbfm.utils.projects.finished_project_data import ProjectData


class TestWormPostureClass(unittest.TestCase):

    def setUp(self):

        # Could use fancier things like parameterized_class, but this is a simple example
        project_list = ["/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml",
                        '/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_wt/2023-09-07_16-11_CaMP7b_O2_worm1-2023-09-07/project_config.yaml']

        # Example project
        project_data = ProjectData.load_final_project_data_from_config(project_list[1], verbose=False)
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

    def test_subsampling_index_deltas(self):
        # Get beh annotation at both high-resolution and fluorescence_fps
        beh_annotation = self.worm.beh_annotation()
        beh_annotation_fluorescence_fps = self.worm.beh_annotation(fluorescence_fps=True)

        # Should only be one index difference
        self.assertTrue(np.unique(np.diff(beh_annotation_fluorescence_fps.index)).size == 1,
                        msg=f"Unique differences: {np.unique(np.diff(beh_annotation_fluorescence_fps.index))}")
        self.assertTrue(np.unique(np.diff(beh_annotation.index)).size == 1,
                        msg=f"Unique differences: {np.unique(np.diff(beh_annotation.index))}")

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
        k3 = "fluorescence_fps"
        kwarg3 = [False, True]

        for v0 in kwarg0:
            for v1 in kwarg1:
                for v2 in kwarg2:
                    for v3 in kwarg3:
                        kwargs = {k0: v0, k1: v1, k2: v2, k3: v3}
                        # print(kwargs)
                        flag = self.worm.worm_speed(**kwargs) is not None
                        if not flag:
                            print(kwargs)
                        self.assertTrue(flag)
