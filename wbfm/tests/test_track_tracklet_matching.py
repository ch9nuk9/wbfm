import unittest
import numpy as np
import pandas as pd


class TestTrackTrackletMatching(unittest.TestCase):

    def setUp(self) -> None:
        self.neuron_name = 'neuron_001'
        self.tracklet_names = ['tracklet_0000000', 'tracklet_0000001']

        tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                              (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.df_tracks = tmp_df

        tmp_df = pd.DataFrame({(self.neuron_name, 'z'): list(range(20)),
                              (self.neuron_name, 'x'): 10*np.array(list(range(20)))})
        self.df_tracks = tmp_df
