import unittest
import numpy as np
import pandas as pd

from wbfm.utils.external.utils_pandas import get_column_name_from_time_and_column_value


class TestTrackTrackletMatching(unittest.TestCase):

    def setUp(self) -> None:
        self.neuron_name = 'neuron_001'
        self.tracklet_names = ['tracklet_0000000', 'tracklet_0000001']

        tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                              (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.df_tracks = tmp_df

        # tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
        #                       (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        # self.df_tracklets = tmp_df

    def test_get_column_name_from_time_and_column_value(self):
        ind, name = get_column_name_from_time_and_column_value(self.df_tracks, 0, 0, 'z')

        self.assertEqual('neuron_001', name)
        self.assertEqual(0, ind)

    def test_no_valid_value(self):
        ind, name = get_column_name_from_time_and_column_value(self.df_tracks, 0, 1, 'z')

        self.assertEqual(None, name)
        self.assertEqual(None, ind)
