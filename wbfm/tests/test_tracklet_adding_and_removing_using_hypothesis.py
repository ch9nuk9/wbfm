import unittest
import random
import hypothesis.strategies as st
from hypothesis import Verbosity, note, event
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, initialize, invariant

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df

# Global variables: the main project object

# project_path = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/project_pytest/project_config.yaml"
# initialization_kwargs = dict(use_custom_padded_dataframe=False, force_tracklets_to_be_sparse=False)
# project_data = ProjectData.load_final_project_data_from_config(project_path,
#                                                                to_load_tracklets=True,
#                                                                to_load_segmentation_metadata=True,
#                                                                to_load_frames=True,
#                                                                initialization_kwargs=initialization_kwargs,
#                                                                verbose=0)
# project_data.use_custom_padded_dataframe = False
# # Load the gui-related tracklet saving objects
# project_data.load_interactive_properties()


class AnnotatorTests(RuleBasedStateMachine):
    # See https://hypothesis.readthedocs.io/en/latest/stateful.html
    def __init__(self):
        super().__init__()
        project_path = "/home/charles/dlc_stacks/project_pytest/project_config.yaml"
        opt = {'log_to_file': False}
        cfg = ModularProjectConfig(project_path, **opt)
        initialization_kwargs = dict(use_custom_padded_dataframe=False, force_tracklets_to_be_sparse=False)
        project_data = ProjectData.load_final_project_data_from_config(cfg,
                                                                       to_load_tracklets=True,
                                                                       to_load_segmentation_metadata=True,
                                                                       to_load_frames=True,
                                                                       initialization_kwargs=initialization_kwargs,
                                                                       verbose=0)
        project_data.use_custom_padded_dataframe = False
        # Load the gui-related tracklet saving objects
        project_data.load_interactive_properties()

        self.project_data = project_data
        self.tracklet_names = get_names_from_df(self.project_data.df_all_tracklets)
        self.neuron_names = self.project_data.neuron_names

        # Set the logging state to not log anything
        self.project_data.logger.setLevel("CRITICAL")

    @rule(data=st.data())
    def test_basic_tracklet_functions(self, data: st.SearchStrategy):
        # Get tracklet name using data strategy
        tracklet_name = data.draw(st.sampled_from(self.tracklet_names))

        # Test functions that don't require a state
        annotator = self.project_data.tracklet_annotator

        # Select a tracklet
        annotator.set_current_tracklet(tracklet_name)
        assert annotator.current_tracklet_name == tracklet_name

        # Clear the current tracklet
        annotator.clear_current_tracklet()
        assert annotator.current_tracklet_name is None
        assert annotator.current_tracklet is None

        previous_tracklet = annotator.previous_tracklet_name
        assert previous_tracklet == tracklet_name

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(data=st.data())
    def test_basic_neuron_functions(self, data: st.SearchStrategy):
        # Get tracklet name using data strategy
        neuron_name = data.draw(st.sampled_from(self.neuron_names))

        # Test functions that don't require a state
        annotator = self.project_data.tracklet_annotator

        # Select a neuron
        annotator.current_neuron = neuron_name
        assert annotator.current_neuron == neuron_name

        # Get the tracklets for the neuron
        tracklets = annotator.get_tracklets_for_neuron(neuron_name)
        assert tracklets is not None
        # Note that there may be no tracklets for a neuron

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(data=st.data())
    def test_already_matched_tracklets(self, data: st.SearchStrategy):

        annotator = self.project_data.tracklet_annotator
        # Get neuron name
        neuron_name = data.draw(st.sampled_from(self.neuron_names))

        # Get a tracklet name from that neuron, if any
        tracklet_dict, current_tracklet, current_name = annotator.get_tracklets_for_neuron(neuron_name)
        # No tracklet should be selected
        assert current_tracklet is None
        assert current_name is None

        # If there are no tracklets, skip this test
        if len(tracklet_dict) == 0:
            return

        # Get an attached tracklet_name, and check properties
        tracklet_name = random.choice(list(tracklet_dict.keys()))
        # Has conflict
        annotator.is_tracklet_already_matched(tracklet_name)
        # Check identity conflict
        name = annotator.get_neuron_name_of_conflicting_match(tracklet_name)
        assert name == neuron_name

        # Select the tracklet and neuron, and check
        annotator.current_neuron = neuron_name
        annotator.set_current_tracklet(tracklet_name)
        assert not annotator.is_current_tracklet_confict_free

        types_of_conflicts = annotator.get_types_of_conflicts()
        assert types_of_conflicts == ["Already added"]

        # Try to add, and check that it didn't work
        flag = annotator.add_tracklet_to_neuron(tracklet_name, neuron_name)
        assert not flag

        # Reset the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_add_and_remove_tracklets(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):
        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))

        # Get a tracklet to add
        tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names))

        # Get the annotator
        annotator = self.project_data.tracklet_annotator

        # Check if the tracklet has conflicts
        has_conflict = annotator.is_tracklet_already_matched(tracklet_name)

        # If conflicts, check that they are found correctly
        if has_conflict:
            # Select the tracklet and neuron, and check
            annotator.set_current_tracklet(tracklet_name)
            annotator.current_neuron = neuron_name

            conflicts = annotator.get_types_of_conflicts()
            assert len(conflicts) > 0
        else:
            # If no conflicts, check that it can be added and removed
            annotator.set_current_tracklet(tracklet_name)
            annotator.current_neuron = neuron_name

            # Add the tracklet
            state_changed = annotator.add_tracklet_to_neuron(tracklet_name, neuron_name)
            assert state_changed

            # Remove the tracklet
            state_changed = annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
            assert state_changed

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(data=st.data())
    def test_remove_and_readd_tracklets_to_neuron(self, data: st.SearchStrategy):
        # Get neuron name
        neuron_name = data.draw(st.sampled_from(self.neuron_names))

        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get the tracklets for the neuron
        tracklets_dict, _, _ = annotator.get_tracklets_for_neuron(neuron_name)
        # If there are no tracklets, skip this test
        if len(tracklets_dict) == 0:
            event(f"Skipping test_remove_and_readd_tracklets_to_neuron because there are no tracklets for {neuron_name}")
            return

        # For each tracklet, remove it and then re-add it
        original_global2neuron_dict = annotator.combined_global2tracklet_dict.copy()
        for tracklet_name in tracklets_dict.keys():
            # Remove the tracklet
            annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
            # Add the tracklet
            annotator.add_tracklet_to_neuron(tracklet_name, neuron_name)
            # Check that the total dictionary for the neuron is the same
            assert annotator.combined_global2tracklet_dict == original_global2neuron_dict

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_add_tracklets(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):

        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))
        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get a tracklet to add; filter for no conflicts and isn't already attached
        # is_not_attached = lambda x: x not in annotator.get_tracklets_for_neuron(neuron_name)[0]
        # is_conflict_free = lambda x: not annotator.is_tracklet_already_matched(x)
        # tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names)).filter(
        #     lambda x: is_not_attached(x) and is_conflict_free(x))
        for i in range(100):
            tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names))
            # Select tracklet and check for conflicts
            annotator.set_current_tracklet(tracklet_name)
            is_conflict_free = not annotator.is_current_tracklet_confict_free
            is_not_attached = tracklet_name not in annotator.get_tracklets_for_neuron(neuron_name)[0]
            if is_conflict_free and is_not_attached:
                break
        else:
            # If no tracklets with no conflicts were found, skip this test
            event("No tracklets with no conflicts were found, skipping test_add_tracklets")
            return

        # Add the tracklet, without removing afterwards
        state_changed = annotator.add_tracklet_to_neuron(tracklet_name, neuron_name)
        assert state_changed

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @rule(data=st.data())
    def test_remove_tracklet_from_neuron(self, data: st.SearchStrategy):
        # Get a neuron to add to
        neuron_name = data.draw(st.sampled_from(self.neuron_names))
        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get the tracklets for the neuron
        tracklets_dict, _, _ = annotator.get_tracklets_for_neuron(neuron_name)
        # If there are no tracklets, skip this test
        if len(tracklets_dict) == 0:
            return

        # Get a tracklet to remove
        tracklet_name = data.draw(st.sampled_from(list(tracklets_dict.keys())))

        # Remove the tracklet
        state_changed = annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
        assert state_changed

        # Deselect the tracklet and neuron
        annotator.clear_current_tracklet()
        annotator.current_neuron = None

    @invariant()
    def nothing_selected(self):
        # Check that nothing is selected
        annotator = self.project_data.tracklet_annotator
        note(f"Current state of the annotator: {annotator.current_status_string()}")
        assert annotator.current_neuron is None
        assert annotator.current_tracklet_name is None
        assert annotator.current_tracklet is None


# Allow the test to be run from the command line
TestAnnotator = AnnotatorTests.TestCase

if __name__ == "__main__":
    AnnotatorTests.TestCase.settings = st.settings(max_examples=2, stateful_step_count=2, verbosity=Verbosity.verbose)
    unittest.main()
