import unittest
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import *

from numpy import array_equal

class TestCropMethods(unittest.TestCase):

    def test_get_crop_coords3d(self):
        crop_sz = (7,5,3)
        center = [10,10,10]
        x, y, z = get_crop_coords3d(center, crop_sz=crop_sz)

        # ground truth
        xt = np.array(range(7,14))
        yt = np.array(range(8,13))
        zt = np.array(range(9,12))

        self.assertTrue(array_equal(x, xt), 'x was wrong')
        self.assertTrue(array_equal(y, yt), 'y was wrong')
        self.assertTrue(array_equal(z, zt), 'z was wrong')

    def test_get_crop_coords3d_no_clipping(self):
        clip_sz = (20,20,20)
        crop_sz = (7,5,3)
        center = [10,10,10]
        x, y, z = get_crop_coords3d(center, crop_sz=crop_sz, clip_sz=clip_sz)

        # ground truth
        xt = np.array(range(7,14))
        yt = np.array(range(8,13))
        zt = np.array(range(9,12))

        self.assertTrue(array_equal(x, xt), 'x was wrong')
        self.assertTrue(array_equal(y, yt), 'y was wrong')
        self.assertTrue(array_equal(z, zt), 'z was wrong')

    def test_get_crop_coords3d_yes_clipping(self):
        clip_sz = (11,11,11)
        crop_sz = (7,5,3)
        center = [10,10,10]
        x, y, z = get_crop_coords3d(center, crop_sz=crop_sz, clip_sz=clip_sz)

        # ground truth
        xt = np.array([7,8,9,10,10,10,10])
        yt = np.array([8,9,10,10,10])
        zt = np.array([9,10,10])

        self.assertTrue(array_equal(x, xt), f"{x} != {xt}")
        self.assertTrue(array_equal(y, yt), f"{y} != {yt}")
        self.assertTrue(array_equal(z, zt), f"{z} != {zt}")

        self.assertEqual(crop_sz[0], len(x))
        self.assertEqual(crop_sz[1], len(y))
        self.assertEqual(crop_sz[2], len(z))



class TestOMECropMethods(unittest.TestCase):

    def test_get_one_frame(self):
        which_z = 7
        num_frames = 2
        sz=(209, 505, 3) # Full frame
        which_neuron = 0
        # mCherry
        folder_name = '/users/charles.fieseler/test_worm1_data/immobilized/'
        fname = 'test_500frames_7slice.ome.tiffDLC_resnet50_Immobilized_mCherryJul24shuffle1_5500.h5'
        annotation_fname = os.path.join(folder_name, fname)

        this_xy, this_prob = xy_from_dlc_dat(annotation_fname, which_neuron=which_neuron, num_frames=num_frames)

        folder_name = '/users/charles.fieseler/test_worm1_data/immobilized/'
        fname = 'test_gcamp_50frames_allslice.ome.tiff'
        video3d_fname = os.path.join(folder_name, fname)

        cropped_dat = get_crop_from_ometiff(video3d_fname, this_xy, which_z, num_frames, sz)

        self.assertEqual(sz+(num_frames,), cropped_dat.shape)


if __name__ == '__main__':
    unittest.main()
