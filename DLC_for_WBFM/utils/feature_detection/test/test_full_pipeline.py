import unittest

import cv2
import numpy as np
import tifffile
import os
import copy
import open3d as o3d
import pandas as pd

from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.feature_detection.visualization_tracks import *
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import *


class TestFullPipeline(unittest.TestCase):

    def setUp(self):
        print("Setting up...")

        # Get the 3d bigtiff folder
        bigtiff_folder = r'D:\More-stabilized-wbfm'
        self.num_slices = 33
        self.alpha = 0.15

        # Will get traces from two videos:
        btf_fname_red = r'test2020-10-22_16-15-20_test4-channel-0-pco_camera1\test2020-10-22_16-15-20_test4-channel-0-pco_camera1bigtiff.btf'
        self.fname = os.path.join(bigtiff_folder, btf_fname_red)

        opt = {'num_slices':self.num_slices, 'alpha':self.alpha}
        vol0 = get_single_volume(btf_fname_red, 0, **opt)
        vol1 = get_single_volume(btf_fname_red, 1, **opt)

        # Detect neurons
        opt = {'num_slices':self.num_slices, 'alpha':1.0, 'verbose':1}
        self.neurons0 = detect_neurons_using_ICP(dat0, **opt)[0]
        self.neurons1 = detect_neurons_using_ICP(dat1, **opt)[0]


        print("Finished setting up.")


    def test_dataset(self):
        # Test
        vol0 = get_single_volume(btf_fname_red, 0, num_slices=self.num_slices, alpha=self.alpha)


    # def


    def test_full_function(self):
        all_matches, all_conf, all_neurons = track_neurons_full_video(self.fname,
                             start_frame=0,
                             num_frames=2,
                             num_slices=self.num_slices,
                             alpha=self.alpha,
                             verbose=0)
