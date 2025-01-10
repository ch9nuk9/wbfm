import unittest

import numpy as np
import pandas as pd

from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.tracklet_class import TrackedWorm, DetectedTrackletsAndNeurons
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once


class TestTrackedWorm(unittest.TestCase):

    def setUp(self) -> None:
        # Build a fake tracklet dataframe
        self.neuron_name = 'neuron_001'
        self.tracklet_names = ['tracklet_0000000', 'tracklet_0000001']

        df_all_tracklets = []
        for name in self.tracklet_names:
            tmp_df = pd.DataFrame({(name, 'z'): list(range(20)),
                                   (name, 'x'): 10*np.array(list(range(20)))})
            df_all_tracklets.append(tmp_df)
        # Concat
        df_all_tracklets = pd.concat(df_all_tracklets, axis=1)

        # Type 1: Conflict at the beginning
        # Add nan at the beginning of the first tracklet, and the end of the second
        df_tracklets = df_all_tracklets.copy()
        df_tracklets.loc[:5, (self.tracklet_names[0], slice(None))] = np.nan
        df_tracklets.loc[8:, (self.tracklet_names[1], slice(None))] = np.nan

        # print(df_tracklets)

        # Build an original match between a neuron and these tracklets
        previous_matches = {self.neuron_name: self.tracklet_names}

        # Build a fake tracklets object
        tracklets_obj = DetectedTrackletsAndNeurons(df_tracklets, None,
                                                    dataframe_output_filename=None,
                                                    use_custom_padded_dataframe=False)

        # Build a TrackedWorm class from fake tracklets and neurons
        worm_obj = TrackedWorm(detections=tracklets_obj, logger=None, verbose=0)
        worm_obj.initialize_neurons_using_previous_matches(previous_matches)

        # Save
        self.worm_obj = worm_obj
        # self.df_all_tracklets = df_tracklets

    def test_init(self):
        pass

    def test_split_tracklet(self):
        worm_obj = self.worm_obj
        df_tracklets = self.worm_obj.detections.df_tracklets_zxy

        split_list_dict = worm_obj.get_conflict_time_dictionary_for_all_neurons()
        df_tracklets_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(df_tracklets, split_list_dict)

        # print(df_tracklets_split)
        # print(name_mapping)

        # Check that the names are correct
        expected_names = ['tracklet_0000002', 'tracklet_0000003', 'tracklet_0000004', 'tracklet_0000005']
        new_names = get_names_from_df(df_tracklets_split)
        self.assertEqual(expected_names, new_names)

        # Check that the number of tracklets is correct
        self.assertEqual(4, len(new_names))

        # Check that the names are mapped correctly
        expected_name_mapping = {'tracklet_0000000': ['tracklet_0000002', 'tracklet_0000003'],
                                 'tracklet_0000001': ['tracklet_0000004', 'tracklet_0000005']}
        self.assertEqual(expected_name_mapping, name_mapping)

        # Check that the nonnan indices are correct
        expected_nonnan_indices = {'tracklet_0000002': [6, 7],
                                   'tracklet_0000003': list(range(8, 20)),
                                   'tracklet_0000004': list(range(6)),
                                   'tracklet_0000005': [6, 7]}
        for name in new_names:
            self.assertEqual(expected_nonnan_indices[name], list(df_tracklets_split[name].dropna().index))
