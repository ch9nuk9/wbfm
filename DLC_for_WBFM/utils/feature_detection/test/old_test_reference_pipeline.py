import unittest

from DLC_for_WBFM.utils.feature_detection.feature_pipeline import *
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import *


class TestReferencePipeline(unittest.TestCase):

    def setUp(self):
        print("Setting up...")

        # Get the 3d bigtiff folder
        bigtiff_folder = r'D:\More-stabilized-wbfm'
        self.num_slices = 33

        btf_fname_red = r'test2020-10-22_16-15-20_test4-channel-0-pco_camera1\test2020-10-22_16-15-20_test4-channel-0-pco_camera1bigtiff.btf'
        self.fname = os.path.join(bigtiff_folder, btf_fname_red)

        print("Finished setting up.")

    def test_pipeline(self):
        track_via_reference_frames(self.fname,
                                   num_slices=self.num_slices,
                                   num_reference_frames=3)
