import unittest
import random
import hypothesis.strategies as st
from hypothesis import given

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


class TestTrackletAddingAndRemoving(unittest.TestCase):

    def setUp(self) -> None:
        # Load a test project
        # See napari_trace_explorer_from_config.py for an example of how to load a project
        project_path = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/project_pytest/project_config.yaml"
        initialization_kwargs = dict(use_custom_padded_dataframe=False,
                                     force_tracklets_to_be_sparse=False,
                                     verbose=0)
        project_data = ProjectData.load_final_project_data_from_config(project_path,
                                                                       to_load_tracklets=True,
                                                                       to_load_segmentation_metadata=True,
                                                                       to_load_frames=True,
                                                                       initialization_kwargs=initialization_kwargs,
                                                                       verbose=0)
        project_data.use_custom_padded_dataframe = False
        # Load the gui-related tracklet saving objects
        project_data.load_interactive_properties()
        self.project_data = project_data
        print("Is verbose?", project_data.verbose)

        # Get names of all objects
        self.tracklet_names = get_names_from_df(project_data.df_all_tracklets)
        self.neuron_names = project_data.neuron_names

    def test_setup(self):
        pass

    def test_basic_tracklet_functions(self):
        # Test functions that don't require a state
        annotator = self.project_data.tracklet_annotator

        # Select a tracklet
        tracklet = self.tracklet_names[0]
        annotator.set_current_tracklet(tracklet)
        self.assertEqual(annotator.current_tracklet_name, tracklet)

        # Clear the current tracklet
        annotator.clear_current_tracklet()
        self.assertIsNone(annotator.current_tracklet_name)
        self.assertIsNone(annotator.current_tracklet)

        previous_tracklet = annotator.previous_tracklet_name
        self.assertEqual(previous_tracklet, tracklet)

    def test_basic_neuron_functions(self):
        # Test functions that don't require a state
        annotator = self.project_data.tracklet_annotator

        # Select a neuron
        neuron = self.neuron_names[0]
        annotator.current_neuron = neuron
        self.assertEqual(annotator.current_neuron, neuron)

        # Get the tracklets for the neuron
        tracklet_dict, current_tracklet, current_name = annotator.get_tracklets_for_neuron(neuron)
        # There is no selected tracklet, so the current tracklet should be None
        self.assertIsNone(current_tracklet)
        self.assertIsNone(current_name)
        # Just check datatype
        self.assertIsInstance(tracklet_dict, dict)

    def test_already_matched_tracklets(self):
        # Test that the tracklets that are already matched are included where they should be
        annotator = self.project_data.tracklet_annotator

        # Select a neuron, but ensure it has tracklets
        selected_neuron = None
        for neuron in self.neuron_names:
            if len(annotator.get_tracklets_for_neuron(neuron)[0]) > 0:
                selected_neuron = neuron
                break
        self.assertIsNotNone(selected_neuron)
        annotator.current_neuron = selected_neuron

        # Get those tracklets
        tracklet_dict, current_tracklet, current_name = annotator.get_tracklets_for_neuron(selected_neuron)
        # No tracklet should be selected
        self.assertIsNone(current_tracklet)
        self.assertIsNone(current_name)
        # But there should be tracklets here
        self.assertGreater(len(tracklet_dict), 0)

        # For each tracklet, check that it has conflicts
        for tracklet_name, _ in tracklet_dict.items():
            # Overall boolean
            self.assertTrue(annotator.is_tracklet_already_matched(tracklet_name))

            # Check identity conflict
            name = annotator.get_neuron_name_of_conflicting_match(tracklet_name)
            self.assertEqual(name, selected_neuron)

            # Check time conflict
            # time_conflict = annotator.tracklet_has_time_overlap(tracklet_name)
            # self.assertTrue(time_conflict)

            # time_conflict_dict = annotator.get_dict_of_tracklet_time_conflicts(tracklet_name)
            # self.assertGreater(len(time_conflict_dict), 0)

            # next_conflict_time, neuron_conflict = annotator.time_of_next_conflict(tracklet_name)
            # self.assertIsNotNone(next_conflict_time)
            # self.assertEqual(neuron_conflict, selected_neuron)

            # Check string of conflicts
            # types_of_conflicts = annotator.get_types_of_conflicts()
            # self.assertEqual(types_of_conflicts, ["Already added"])

            # Select the tracklet, and check
            annotator.set_current_tracklet(tracklet_name)
            self.assertFalse(annotator.is_current_tracklet_confict_free)

            # Try to add, and check that it didn't work
            flag = annotator._add_tracklet_to_neuron(tracklet_name, selected_neuron)
            self.assertFalse(flag)

    def test_add_and_remove_tracklets(self):
        # Test that the tracklets that are already matched are included where they should be
        annotator = self.project_data.tracklet_annotator

        # Select a neuron, but ensure it has tracklets
        selected_neuron = None
        for neuron in self.neuron_names:
            if len(annotator.get_tracklets_for_neuron(neuron)[0]) > 0:
                selected_neuron = neuron
                break
        self.assertIsNotNone(selected_neuron)
        annotator.current_neuron = selected_neuron

        # Get those tracklets
        tracklet_dict, current_tracklet, current_name = annotator.get_tracklets_for_neuron(selected_neuron)
        # No tracklet should be selected
        self.assertIsNone(current_tracklet)
        self.assertIsNone(current_name)
        # But there should be tracklets here
        self.assertGreater(len(tracklet_dict), 0)

        # Select a tracklet that is conflict free
        tracklet_name = None
        for i in range(1000):
            tracklet_name = self._get_random_tracklet()
            annotator.set_current_tracklet(tracklet_name)
            if annotator.is_current_tracklet_confict_free:
                break
        self.assertIsNotNone(tracklet_name)
        annotator.set_current_tracklet(tracklet_name)

        # Add the tracklet
        flag = annotator._add_tracklet_to_neuron(tracklet_name, selected_neuron)
        self.assertTrue(flag)

        # Remove the tracklet
        flag = annotator.remove_tracklet_from_neuron(tracklet_name, selected_neuron)
        self.assertTrue(flag)

        # Deselect the tracklet
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    def test_remove_and_readd_tracklets_to_neuron(self):
        # Get neuron name
        neuron_name = self._get_random_neuron()

        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get the tracklets for the neuron
        tracklets_dict, _, _ = annotator.get_tracklets_for_neuron(neuron_name)
        # If there are no tracklets, skip this test
        if len(tracklets_dict) == 0:
            return

        # For each tracklet, remove it and then re-add it
        original_global2neuron_dict = annotator.combined_global2tracklet_dict.copy()
        for tracklet_name in tracklets_dict.keys():
            # Remove the tracklet
            annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
            # Add the tracklet
            annotator._add_tracklet_to_neuron(tracklet_name, neuron_name)
            # Check that the total dictionary for the neuron is the same
            assert annotator.combined_global2tracklet_dict == original_global2neuron_dict

    # Functions for hypothesis testing
    def _get_random_tracklet(self):
        return random.choice(self.tracklet_names)

    # Use for hypothesis testing
    # @given(tracklet_name=st.sampled_from(tracklet_names))
    # def test_hypothesis_tracklet_functions(self, tracklet_name):
    #     # Test functions that don't require a state
    #     annotator = self.project_data.tracklet_annotator
    #
    #     # Select a tracklet
    #     annotator.set_current_tracklet(tracklet_name)
    #     self.assertEqual(annotator.current_tracklet_name, tracklet_name)
    #
    #     # Clear the current tracklet
    #     annotator.clear_current_tracklet()
    #     self.assertIsNone(annotator.current_tracklet_name)
    #     self.assertIsNone(annotator.current_tracklet)
    #
    #     previous_tracklet = annotator.previous_tracklet_name
    #     self.assertEqual(previous_tracklet, tracklet_name)

    def _get_random_tracklet_attached_to_current_neuron(self):
        return random.choice(self.project_data.tracklet_annotator.get_tracklets_for_neuron()[0])

    def _get_random_neuron(self):
        return random.choice(self.neuron_names)

    def _get_random_tracklet_not_attached_to_current_neuron(self):
        # Get the current neuron
        current_neuron = self.project_data.tracklet_annotator.current_neuron
        # Get all tracklets
        all_tracklets = self.tracklet_names
        # Get all tracklets attached to the current neuron
        attached_tracklets = self.project_data.tracklet_annotator.get_tracklets_for_neuron()[0]
        # Get all tracklets not attached to the current neuron
        not_attached_tracklets = list(set(all_tracklets) - set(attached_tracklets))
        # Return a random one
        return random.choice(not_attached_tracklets)
