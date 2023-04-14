import unittest
import random
import hypothesis.strategies as st
from hypothesis import Verbosity, note, event, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from wbfm.utils.external.utils_pandas import get_times_of_conflicting_dataframes
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


class AnnotatorTests(RuleBasedStateMachine):
    # See https://hypothesis.readthedocs.io/en/latest/stateful.html
    def __init__(self):
        super().__init__()
        project_path = "/home/charles/dlc_stacks/project_pytest/project_config.yaml"
        opt = {'log_to_file': False}
        cfg = ModularProjectConfig(project_path, **opt)
        initialization_kwargs = dict(use_custom_padded_dataframe=False, force_tracklets_to_be_sparse=False,
                                     verbose=0)
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
        self.original_tracklet_names = self.tracklet_names.copy()
        self.neuron_names = self.project_data.neuron_names

        self.expected_number_of_tracklets = len(self.tracklet_names)

        # Set the logging state to not log anything
        self.project_data.logger.setLevel("CRITICAL")
        self.project_data.tracklet_annotator.logger.setLevel("CRITICAL")
        self.project_data.tracklet_annotator.verbose = 0

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
        annotator.clear_tracklet_and_neuron()

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
            event(f"Skipping test for neuron {neuron_name} because it has no tracklets")
            assume(len(tracklet_dict) > 0)

            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # Get an attached tracklet_name, and check properties
        tracklet_name = random.choice(list(tracklet_dict.keys()))
        assert annotator.is_tracklet_already_matched(tracklet_name)
        # Must have a conflict: the tracklet is already matched to our target neuron
        name = annotator.get_neuron_name_of_conflicting_match(tracklet_name)
        note(f"{tracklet_name} should be in: {list(tracklet_dict.keys())}")
        note(f"Conflicting neuron {name} should be the same neuron: "
             f"{list(annotator.combined_global2tracklet_dict[name])}")
        assert name == neuron_name

        # Select the tracklet and neuron, and check
        annotator.current_neuron = neuron_name
        annotator.set_current_tracklet(tracklet_name)
        assert not annotator.is_current_tracklet_confict_free

        types_of_conflicts = annotator.get_types_of_conflicts()
        assert types_of_conflicts == ["Already added"]

        # Try to add, and check that it didn't work
        flag = annotator.save_current_tracklet_to_current_neuron()
        assert not flag

        # Reset the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_add_and_remove_tracklets(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):
        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))

        # Get a tracklet to add
        tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names))

        # Get the annotator
        annotator = self.project_data.tracklet_annotator

        # Check if the tracklet has conflicts
        annotator.set_current_tracklet(tracklet_name)
        annotator.current_neuron = neuron_name
        conflict_free = annotator.is_current_tracklet_confict_free

        # Get the current state of the neuron for later checking
        original_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])

        # If conflicts, check that they are found correctly
        if not conflict_free:
            conflicts = annotator.get_types_of_conflicts()
            assert len(conflicts) > 0
            note(f"Not adding {tracklet_name} to {neuron_name} because of conflicts: {conflicts}")
        else:
            # Add the tracklet
            state_changed = annotator.save_current_tracklet_to_current_neuron()
            assert state_changed

            # Remove the tracklet
            state_changed = annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
            assert state_changed

        # Check that the total dictionary for the neuron is the same
        # But ignore the order of the tracklets
        new_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
        assert new_tracklet_set == original_tracklet_set
        note(f"Added and removed tracklet {tracklet_name} from neuron {neuron_name} successfully")

        # Deselect the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

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
            note(f"Skipping test_remove_and_readd_tracklets_to_neuron because there are no tracklets for {neuron_name}")
            assume(len(tracklets_dict) > 0)
            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # For each tracklet, remove it and then re-add it
        original_global2neuron_dict = annotator.combined_global2tracklet_dict[neuron_name]
        for tracklet_name in tracklets_dict.keys():
            # Remove the tracklet
            annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
            # Add the tracklet; requires selecting it first
            annotator.set_current_tracklet(tracklet_name)
            annotator.save_current_tracklet_to_current_neuron()
            # Check that the total dictionary for the neuron is the same
            # But ignore the order of the tracklets
            new_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
            original_tracklet_set = set(original_global2neuron_dict)
            assert new_tracklet_set == original_tracklet_set
            note(f"Removed and re-added tracklet {tracklet_name} to neuron {neuron_name} successfully")

        # Deselect the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

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
            is_conflict_free = annotator.is_current_tracklet_confict_free
            is_not_attached = tracklet_name not in annotator.get_tracklets_for_neuron(neuron_name)[0]
            assume(is_not_attached and is_conflict_free)
            if is_conflict_free and is_not_attached:
                note(f"Found tracklet {tracklet_name} with no conflicts and not attached to {neuron_name}")
                break
        else:
            # If no tracklets with no conflicts were found, skip this test
            note("No tracklets with no conflicts were found, skipping test_add_tracklets")

            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # Add the tracklet, without removing afterwards
        original_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
        state_changed = annotator.save_current_tracklet_to_current_neuron()
        assert state_changed

        # Check that the total dictionary for the neuron not is the same
        new_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
        assert new_tracklet_set == original_tracklet_set | {tracklet_name}

        # Deselect the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_remove_tracklet_from_neuron(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):
        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))
        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get the tracklets for the neuron
        tracklets_dict, _, _ = annotator.get_tracklets_for_neuron(neuron_name)
        if len(tracklets_dict) == 0:
            note(f"Skipping test_remove_tracklet_from_neuron because there are no tracklets for {neuron_name}")
            assume(len(tracklets_dict) > 0)
            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # Get a tracklet to remove
        tracklet_name = tracklet_data.draw(st.sampled_from(list(tracklets_dict.keys())))

        # Remove the tracklet
        original_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
        state_changed = annotator.remove_tracklet_from_neuron(tracklet_name, neuron_name)
        assert state_changed

        # Check that the total dictionary for the neuron is not the same
        # But ignore the order of the tracklets
        new_tracklet_set = set(annotator.combined_global2tracklet_dict[neuron_name])
        assert new_tracklet_set == original_tracklet_set - {tracklet_name}

        # Deselect the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_split_tracklet_and_add(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):

        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))
        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get a tracklet to add; filter for time conflict and isn't already attached
        for i in range(100):
            tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names))
            # Select tracklet and check for conflicts
            annotator.set_current_tracklet(tracklet_name)
            these_tracklets, current_tracklet, current_tracklet_name = annotator.get_tracklets_for_neuron()
            assume(annotator.tracklet_has_time_overlap() and not annotator.is_current_tracklet_confict_free and
                   tracklet_name not in these_tracklets and len(current_tracklet.dropna().index) > 1)
            # Do I need this check, or does the assume take care of it?
            if annotator.is_current_tracklet_confict_free:
                continue
            elif tracklet_name in these_tracklets:
                continue
            elif len(current_tracklet.dropna().index) > 1:
                continue
            elif annotator.tracklet_has_time_overlap():
                note(f"Found tracklet {tracklet_name} with time conflicts and not attached to {neuron_name}")
                break
        else:
            # If no tracklets with no conflicts were found, skip this test
            note("No tracklets with no conflicts were found, skipping test_add_tracklets")

            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # Check for identity conflicts, and remove if there are any
        conflict_types = annotator.get_types_of_conflicts()
        if "Identity" in conflict_types:
            annotator.remove_tracklet_from_all_matches()
            note("First, fixed identity conflicts")
        assert "Time" in conflict_types

        # Check the time conflict points, and split the tracklet there
        original_tracklet_name = annotator.current_tracklet_name
        new_tracklet_names = []
        # Print all the time conflicts
        conflicts = annotator.get_dict_of_tracklet_time_conflicts()
        note(f"All initial time conflicts: {conflicts}")

        # Get the times of the conflicts from the actual tracklets
        these_tracklets, current_tracklet, current_tracklet_name = annotator.get_tracklets_for_neuron()
        note(f"Current tracklet: {current_tracklet}")
        note(f"Already attached tracklets: {these_tracklets}")
        tracklet_list = list(these_tracklets.values())
        tracklet_list.append(current_tracklet)
        tracklet_names = list(these_tracklets.keys())
        tracklet_names.append(current_tracklet_name)

        overlapping_tracklet_conflict_points = get_times_of_conflicting_dataframes(tracklet_list,
                                                                                   tracklet_names,
                                                                                   verbose=0)
        note(f"Overlapping tracklet conflict points: {overlapping_tracklet_conflict_points}")
        for tracklet_name, split_list in overlapping_tracklet_conflict_points.items():
            # Split the tracklet so that no more conflicts are possible
            annotator.set_current_tracklet(tracklet_name)
            split_list.sort()
            for t in split_list:
                # Split and keep the new half, because the list is sorted
                successfully_split = annotator.split_current_tracklet(t, set_right_half_to_current=True, verbose=0)
                assert successfully_split
                self.expected_number_of_tracklets += 1
                # Remove the old tracklet from the neuron
                previous_tracklet_name = annotator.previous_tracklet_name
                annotator.remove_tracklet_from_neuron(previous_tracklet_name)

                new_tracklet_names.append(annotator.current_tracklet_name)

        note(f"Generated {len(new_tracklet_names)} new tracklets from splits: {new_tracklet_names}")

        # Try to attach part of the split tracklet, and check that one worked
        original_tracklet_attached = annotator.save_current_tracklet_to_current_neuron()
        for new_tracklet_name in new_tracklet_names:
            note(f"Attempting to attach new tracklet: {new_tracklet_name}")
            annotator.set_current_tracklet(new_tracklet_name)
            new_tracklet_attached = annotator.save_current_tracklet_to_current_neuron()
            if new_tracklet_attached:
                break
        else:
            assert original_tracklet_attached

        # Deselect the tracklet and neuron
        annotator.clear_tracklet_and_neuron()

    @rule(neuron_data=st.data(), tracklet_data=st.data())
    def test_remove_conflicts_and_add_tracklet(self, neuron_data: st.SearchStrategy, tracklet_data: st.SearchStrategy):

        # Get a neuron to add to
        neuron_name = neuron_data.draw(st.sampled_from(self.neuron_names))
        # Get the annotator
        annotator = self.project_data.tracklet_annotator
        annotator.current_neuron = neuron_name

        # Get a tracklet to add with conflicts
        for i in range(100):
            tracklet_name = tracklet_data.draw(st.sampled_from(self.tracklet_names))
            # Select tracklet and check for conflicts
            annotator.set_current_tracklet(tracklet_name)
            is_conflict_free = annotator.is_current_tracklet_confict_free
            assume(not is_conflict_free)
            if not is_conflict_free:
                note(f"Found tracklet {tracklet_name} with conflicts")
                break
        else:
            # If no tracklets with no conflicts were found, skip this test
            note("No tracklets with no conflicts were found, skipping test_add_tracklets")

            # Deselect the tracklet and neuron
            annotator.clear_tracklet_and_neuron()
            return

        # Remove the conflicts
        types_of_conflicts = annotator.get_types_of_conflicts()
        assert len(types_of_conflicts) > 0
        assert types_of_conflicts[0] != "No conflicts"
        conflict_removal_dict = {"Already added": 'remove_tracklet_from_all_matches',
                                 "Identity": 'remove_tracklet_from_all_matches',
                                 "Time": 'remove_tracklets_with_time_conflicts'}
        for conflict_type in types_of_conflicts:
            # Perform the conflict removal
            conflict_removal_function = getattr(annotator, conflict_removal_dict[conflict_type])
            conflict_removal_function()

        # Check that the conflicts are gone
        assert annotator.is_current_tracklet_confict_free

        # Add the tracklet to the neuron
        state_changed = annotator.save_current_tracklet_to_current_neuron()
        assert state_changed

        # Deselect
        annotator.clear_tracklet_and_neuron()

    @invariant()
    def test_nothing_selected(self):
        # Check that nothing is selected
        annotator = self.project_data.tracklet_annotator
        # note(f"Current state of the annotator: {annotator.current_status_string()}")
        assert annotator.current_neuron is None
        assert annotator.current_tracklet_name is None
        assert annotator.current_tracklet is None

    @invariant()
    def test_number_of_tracklets(self):
        # Check that the number of tracklets is correct
        annotator = self.project_data.tracklet_annotator
        assert annotator.df_tracklet_obj.num_total_tracklets == self.expected_number_of_tracklets

    @invariant()
    def test_original_tracklet_names(self):
        # Check that all the original tracklet names are a subset of the current names
        original_names = self.original_tracklet_names
        current_names = self.project_data.tracklet_annotator.df_tracklet_obj.all_tracklet_names
        assert set(original_names).issubset(set(current_names))


# Allow the test to be run from the command line
TestAnnotator = AnnotatorTests.TestCase

if __name__ == "__main__":
    AnnotatorTests.TestCase.settings = st.settings(max_examples=2, stateful_step_count=2, verbosity=Verbosity.verbose)
    unittest.main()
