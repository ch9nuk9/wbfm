# Display more informative error messages
# https://www.tutorialexample.com/fix-pyqt-gui-application-crashed-while-no-error-message-displayed-a-beginner-guide-pyqt-tutorial/
import cgitb
import os
import signal

from DLC_for_WBFM.gui.utils.utils_matplotlib import PlotQWidget
from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df
from DLC_for_WBFM.utils.projects.utils_project_status import check_all_needed_data_for_step

cgitb.enable(format='text')
import logging
from PyQt5.QtWidgets import QApplication
logger = logging.getLogger('traceExplorerLogger')
logger.setLevel(logging.INFO)
import sys
import napari
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from DLC_for_WBFM.gui.utils.utils_gui import zoom_using_layer_in_viewer, change_viewer_time_point, \
    build_tracks_from_dataframe, zoom_using_viewer
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


class NapariTraceExplorer(QtWidgets.QWidget):

    subplot_is_initialized = False
    tracklet_lines = None
    zoom_opt = None
    subplot_xlim = None

    _disable_callbacks = False

    def __init__(self, project_data: ProjectData, app: QApplication):
        check_all_needed_data_for_step(project_data.project_config.self_path,
                                       step_index=5, raise_error=True, training_data_required=False)

        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dat = project_data
        self.main_window = app

        self.tracklet_lines = []
        self.subplot_xlim = []
        self.zoom_opt = {'zoom': None, 'ind_within_layer': 0, 'layer_is_full_size_and_single_neuron': False,
                         'layer_name': 'final_track'}
        logger.info("Finished initializing Trace Explorer object")

    def setupUi(self, viewer: napari.Viewer):

        logger.info("Starting main UI setup")
        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = get_names_from_df(self.dat.red_traces)
        self.current_name = neuron_names[0]

        # BOX 1: Change neurons (dropdown)
        self.groupBox1 = QtWidgets.QGroupBox("Neuron selection", self.verticalLayoutWidget)
        self.vbox1 = QtWidgets.QVBoxLayout(self.groupBox1)
        self.changeNeuronsDropdown = QtWidgets.QComboBox()
        self.changeNeuronsDropdown.addItems(neuron_names)
        self.changeNeuronsDropdown.setItemText(0, self.current_name)
        self.changeNeuronsDropdown.currentIndexChanged.connect(self.change_neurons)
        # self.verticalLayout.addWidget(self.changeNeuronsDropdown)
        self.vbox1.addWidget(self.changeNeuronsDropdown)

        # BOX 2: overall mode options
        # Change traces (dropdown)
        self.groupBox2 = QtWidgets.QGroupBox("Channel and Mode selection", self.verticalLayoutWidget)
        self.vbox2 = QtWidgets.QVBoxLayout(self.groupBox2)

        self.changeChannelDropdown = QtWidgets.QComboBox()
        self.changeChannelDropdown.addItems(['green', 'red', 'ratio'])
        self.changeChannelDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox2.addWidget(self.changeChannelDropdown)

        # Change traces vs tracklet mode
        self.changeTraceTrackletDropdown = QtWidgets.QComboBox()
        self.changeTraceTrackletDropdown.addItems(['traces', 'tracklets'])
        self.changeTraceTrackletDropdown.currentIndexChanged.connect(self.change_trace_tracklet_mode)
        self.vbox2.addWidget(self.changeTraceTrackletDropdown)

        self.changeInteractivityCheckbox = QtWidgets.QCheckBox("Turn on interactivity? "
                                                               "NOTE: only Raw_segmentation layer is interactive")
        self.changeInteractivityCheckbox.stateChanged.connect(self.update_interactivity)
        self.vbox2.addWidget(self.changeInteractivityCheckbox)

        # More complex boxes:

        self._setup_trace_filtering_buttons()  # Box 3
        self._setup_general_shortcut_buttons()
        self._setup_tracklet_correction_shortcut_buttons()  # Box 4
        self._setup_segmentation_correction_buttons()  # Box 5

        self.verticalLayout.addWidget(self.groupBox1)
        self.verticalLayout.addWidget(self.groupBox2)
        self.verticalLayout.addWidget(self.groupBox3)
        self.verticalLayout.addWidget(self.groupBox3b)
        self.verticalLayout.addWidget(self.groupBox4)
        self.verticalLayout.addWidget(self.groupBox5)

        self.initialize_track_layers()
        self.initialize_shortcuts()
        self.initialize_trace_or_tracklet_subplot()
        self.update_interactivity()

        logger.info("Finished main UI setup")

    def _setup_trace_filtering_buttons(self):
        # Change traces (dropdown)
        self.groupBox3 = QtWidgets.QGroupBox("Trace calculation options", self.verticalLayoutWidget)
        self.formlayout3 = QtWidgets.QFormLayout(self.groupBox3)

        self.changeTraceCalculationDropdown = QtWidgets.QComboBox()
        self.changeTraceCalculationDropdown.addItems(['integration', 'z', 'volume'])
        # , 'likelihood' ... Too short
        self.changeTraceCalculationDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Trace calculation (y axis):", self.changeTraceCalculationDropdown)
        # Change trace filtering (checkbox)
        self.changeTraceFilteringDropdown = QtWidgets.QComboBox()
        self.changeTraceFilteringDropdown.addItems(['no_filtering', 'rolling_mean', 'linear_interpolation'])
        self.changeTraceFilteringDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Trace filtering:", self.changeTraceFilteringDropdown)
        # Change trace outlier removal (dropdown)
        # self.changeTraceOutlierCheckBox = QtWidgets.QCheckBox()
        # self.changeTraceOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        # self.formlayout3.addRow("Remove outliers (activity)?", self.changeTraceOutlierCheckBox)

        # self.changeTrackingOutlierCheckBox = QtWidgets.QCheckBox()
        # self.changeTrackingOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        # self.formlayout3.addRow("Remove outliers (tracking confidence)?", self.changeTrackingOutlierCheckBox)

        # TODO: spin box must be integers
        # self.changeTrackingOutlierSpinBox = QtWidgets.QSpinBox()
        # self.changeTrackingOutlierSpinBox.setRange(0, 1)
        # self.changeTrackingOutlierSpinBox.setSingleStep(0.1)
        # self.changeTrackingOutlierSpinBox.valueChanged.connect(self.update_trace_subplot)
        # self.formlayout3.addRow("Outlier threshold:", self.changeTrackingOutlierSpinBox)

    def _setup_general_shortcut_buttons(self):
        self.groupBox3b = QtWidgets.QGroupBox("General shortcuts", self.verticalLayoutWidget)
        self.vbox3b = QtWidgets.QVBoxLayout(self.groupBox3b)

        # self.refreshButton = QtWidgets.QPushButton("Refresh Subplot (r)")
        # self.refreshButton.pressed.connect(self.update_trace_or_tracklet_subplot)
        # self.vbox3b.addWidget(self.refreshButton)

        self.refreshDefaultLayersButton = QtWidgets.QPushButton("Refresh Default Napari Layers")
        self.refreshDefaultLayersButton.pressed.connect(self.refresh_default_napari_layers)
        self.vbox3b.addWidget(self.refreshDefaultLayersButton)

        self.printTrackletsButton = QtWidgets.QPushButton("Print tracklets attached to current neuron (v)")
        self.printTrackletsButton.pressed.connect(self.print_tracklets)
        self.vbox3b.addWidget(self.printTrackletsButton)

        # self.zoom1Button = QtWidgets.QPushButton("Next time point (d)")
        # self.zoom1Button.pressed.connect(self.zoom_next)
        # self.vbox3b.addWidget(self.zoom1Button)
        # self.zoom2Button = QtWidgets.QPushButton("Previous time point (a)")
        # self.zoom2Button.pressed.connect(self.zoom_previous)
        # self.vbox3b.addWidget(self.zoom2Button)
        self.zoom3Button = QtWidgets.QPushButton("Zoom to next time with nan (f)")
        self.zoom3Button.pressed.connect(self.zoom_to_next_nan)
        self.vbox3b.addWidget(self.zoom3Button)

    def _setup_tracklet_correction_shortcut_buttons(self):
        # BOX 4: tracklet shortcuts
        self.groupBox4 = QtWidgets.QGroupBox("Tracklet Correction", self.verticalLayoutWidget)
        self.vbox4 = QtWidgets.QVBoxLayout(self.groupBox4)

        self.trackletHint1 = QtWidgets.QLabel("Normal Click: Select tracklet attached to neuron")
        self.vbox4.addWidget(self.trackletHint1)

        self.recentTrackletSelector = QtWidgets.QComboBox()
        self.vbox4.addWidget(self.recentTrackletSelector)
        self.recentTrackletSelector.currentIndexChanged.connect(self.change_tracklets_using_dropdown)
        self.recentTrackletSelector.setToolTip("Select from history of recent tracklets")

        self.zoom4Button = QtWidgets.QPushButton("Zoom to next time with tracklet conflict (g)")
        self.zoom4Button.pressed.connect(self.zoom_to_next_conflict)
        self.zoom4Button.setToolTip("Note: does nothing if there is no tracklet selected")
        self.vbox4.addWidget(self.zoom4Button)
        self.zoom5Button = QtWidgets.QPushButton("Zoom to end of current tracklet (j)")
        self.zoom5Button.setToolTip("Alternative: zoom to beginning of current tracklet (h)")
        self.zoom5Button.pressed.connect(self.zoom_to_end_of_current_tracklet)
        self.vbox4.addWidget(self.zoom5Button)

        self.toggleSegButton = QtWidgets.QPushButton("Toggle Raw segmentation layer (s)")
        self.toggleSegButton.pressed.connect(self.toggle_raw_segmentation_layer)
        self.splitTrackletButton1 = QtWidgets.QPushButton("Split current tracklet (keep left) (q)")
        self.splitTrackletButton1.pressed.connect(self.split_current_tracklet_keep_left)
        self.vbox4.addWidget(self.splitTrackletButton1)
        self.splitTrackletButton2 = QtWidgets.QPushButton("Split current tracklet (keep right) (e)")
        self.splitTrackletButton2.pressed.connect(self.split_current_tracklet_keep_right)
        self.vbox4.addWidget(self.splitTrackletButton2)
        self.clearTrackletButton = QtWidgets.QPushButton("Clear current tracklet (w)")
        self.clearTrackletButton.pressed.connect(self.clear_current_tracklet)
        self.vbox4.addWidget(self.clearTrackletButton)
        self.removeTrackletButton1 = QtWidgets.QPushButton("Remove tracklets with time conflicts")
        self.removeTrackletButton1.pressed.connect(self.remove_time_conflicts)
        self.vbox4.addWidget(self.removeTrackletButton1)
        self.removeTrackletButton2 = QtWidgets.QPushButton("Remove current tracklet from all neurons")
        self.removeTrackletButton2.pressed.connect(self.remove_tracklet_from_all_matches)
        self.vbox4.addWidget(self.removeTrackletButton2)
        self.appendTrackletButton = QtWidgets.QPushButton("Save current tracklet to neuron (IF conflict-free) (c)")
        self.appendTrackletButton.pressed.connect(self.save_current_tracklet_to_neuron)
        self.appendTrackletButton.setToolTip("Note: check console for more details of the conflict")
        self.vbox4.addWidget(self.appendTrackletButton)

        self.saveSegmentationToTrackletButton = QtWidgets.QPushButton("Save current segmentation "
                                                                      "to current tracklet (x)")
        self.saveSegmentationToTrackletButton.pressed.connect(self.save_segmentation_to_tracklet)
        self.vbox4.addWidget(self.saveSegmentationToTrackletButton)

        self.saveTrackletsStatusLabel = QtWidgets.QLabel("STATUS: No tracklet loaded")
        self.vbox4.addWidget(self.saveTrackletsStatusLabel)

        self.list_of_tracklet_correction_widgets = [
            self.recentTrackletSelector,
            self.zoom4Button,
            self.zoom5Button,
            self.toggleSegButton,
            self.splitTrackletButton1,
            self.splitTrackletButton2,
            self.clearTrackletButton,
            self.removeTrackletButton1,
            self.removeTrackletButton2,
            self.appendTrackletButton,
            self.saveSegmentationToTrackletButton
        ]

    def _setup_segmentation_correction_buttons(self):
        self.groupBox5 = QtWidgets.QGroupBox("Segmentation Correction", self.verticalLayoutWidget)
        self.formlayout5 = QtWidgets.QFormLayout(self.groupBox5)

        self.splitSegmentationHint1 = QtWidgets.QLabel()
        self.splitSegmentationHint1.setText("Select segmentation")
        self.formlayout5.addRow("Control-click:", self.splitSegmentationHint1)
        self.splitSegmentationHint2 = QtWidgets.QLabel()
        self.splitSegmentationHint2.setText("Select segmentation and try to split")
        self.formlayout5.addRow("Alt-click:", self.splitSegmentationHint2)

        self.splitSegmentationManualSliceButton = QtWidgets.QSpinBox()
        self.splitSegmentationManualSliceButton.setRange(1, 20)  # TODO: look at actual z depth of neuron
        self.splitSegmentationManualSliceButton.valueChanged.connect(self.update_segmentation_options)
        self.formlayout5.addRow("Manual slice index: ", self.splitSegmentationManualSliceButton)

        self.splitSegmentationKeepOriginalIndexButton = QtWidgets.QComboBox()
        self.splitSegmentationKeepOriginalIndexButton.addItems(["Top", "Bottom"])
        self.splitSegmentationKeepOriginalIndexButton.currentIndexChanged.connect(self.update_segmentation_options)
        self.formlayout5.addRow("Which side keeps original index: ", self.splitSegmentationKeepOriginalIndexButton)

        self.clearSelectedSegmentationsButton = QtWidgets.QPushButton("Clear")
        self.clearSelectedSegmentationsButton.pressed.connect(self.clear_current_segmentations)
        self.formlayout5.addRow("Remove selected segmentations: ", self.clearSelectedSegmentationsButton)

        self.splitSegmentationManualButton = QtWidgets.QPushButton("Try to manually split")
        self.splitSegmentationManualButton.pressed.connect(self.split_segmentation_manual)
        self.formlayout5.addRow("Produce candidate mask: ", self.splitSegmentationManualButton)
        self.splitSegmentationAutomaticButton = QtWidgets.QPushButton("Try to automatically split")
        self.splitSegmentationAutomaticButton.pressed.connect(self.split_segmentation_automatic)
        self.formlayout5.addRow("Produce candidate mask: ", self.splitSegmentationAutomaticButton)
        self.mergeSegmentationButton = QtWidgets.QPushButton("Try to merge selected")
        self.mergeSegmentationButton.pressed.connect(self.merge_segmentation)
        self.formlayout5.addRow("Produce candidate mask: ", self.mergeSegmentationButton)

        self.splitSegmentationSaveButton1 = QtWidgets.QPushButton("Save to RAM")
        self.splitSegmentationSaveButton1.pressed.connect(self.modify_segmentation_using_manual_correction)
        self.formlayout5.addRow("Save candidate mask: ", self.splitSegmentationSaveButton1)
        self.mainSaveButton = QtWidgets.QPushButton("SAVE TO DISK")
        self.mainSaveButton.pressed.connect(self.modify_segmentation_and_tracklets_on_disk)
        self.formlayout5.addRow("***Masks and Tracklets***", self.mainSaveButton)

        self.saveSegmentationStatusLabel = QtWidgets.QLabel("No segmentation loaded")
        self.formlayout5.addRow("STATUS: ", self.saveSegmentationStatusLabel)

        self.update_segmentation_options()

        self.list_of_segmentation_correction_widgets = [
            self.splitSegmentationManualSliceButton,
            self.splitSegmentationKeepOriginalIndexButton,
            self.clearSelectedSegmentationsButton,
            self.splitSegmentationManualButton,
            self.splitSegmentationAutomaticButton,
            self.mergeSegmentationButton,
            self.splitSegmentationSaveButton1,
            self.mainSaveButton,
        ]

    @property
    def seg_layer(self):
        return self.viewer.layers['Raw segmentation']

    def refresh_default_napari_layers(self):
        self.dat.add_layers_to_viewer(self.viewer, which_layers='all', check_if_layers_exist=True)

    def change_neurons(self):
        if not self._disable_callbacks:
            self.update_dataframe_using_points()
            self.update_neuron_in_tracklet_annotator()
            self.update_track_layers()
            self.update_trace_or_tracklet_subplot()

    def change_tracklets_using_dropdown(self):
        if not self._disable_callbacks:
            self.dat.tracklet_annotator.set_current_tracklet(self.recentTrackletSelector.currentText())
            self.dat.tracklet_annotator.add_current_tracklet_to_viewer(self.viewer)
            self.tracklet_updated_psuedo_event()

    def add_to_recent_tracklet_dropdown(self):
        last_tracklet = self.dat.tracklet_annotator.current_tracklet_name
        if last_tracklet is None:
            return
        current_items = [self.recentTrackletSelector.itemText(i) for i in range(self.recentTrackletSelector.count())]

        if last_tracklet in current_items:
            return

        self._disable_callbacks = True
        self.recentTrackletSelector.insertItem(0, last_tracklet)

        num_to_remember = 8
        if self.recentTrackletSelector.count() > num_to_remember:
            self.recentTrackletSelector.removeItem(num_to_remember)
        self._disable_callbacks = False

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['final_track'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)

    def update_neuron_in_tracklet_annotator(self):
        self.dat.tracklet_annotator.current_neuron = self.changeNeuronsDropdown.currentText()

    def update_segmentation_options(self):
        self.dat.tracklet_annotator.segmentation_options = dict(
            which_neuron_keeps_original=self.splitSegmentationKeepOriginalIndexButton.currentText(),
            # method=self.splitSegmentationMethodButton.currentText(),
            x_split_local_coord=self.splitSegmentationManualSliceButton.value()
        )

    def update_interactivity(self):
        to_be_interactive = self.changeInteractivityCheckbox.isChecked()
        self.dat.tracklet_annotator.is_currently_interactive = to_be_interactive

        if to_be_interactive:
            self.groupBox4.setTitle("Tracklet Correction (currently enabled)")
            self.groupBox5.setTitle("Segmentation Correction (currently enabled)")
        else:
            self.groupBox4.setTitle("Tracklet Correction (currently disabled)")
            self.groupBox5.setTitle("Segmentation Correction (currently disabled)")

        for widget in self.list_of_segmentation_correction_widgets:
            widget.setEnabled(to_be_interactive)

        for widget in self.list_of_tracklet_correction_widgets:
            widget.setEnabled(to_be_interactive)

    def modify_segmentation_using_manual_correction(self):
        # Uses candidate mask layer
        self.dat.modify_segmentation_using_manual_correction()
        self.dat.tracklet_annotator.update_segmentation_layer_using_buffer(self.seg_layer)
        self.dat.tracklet_annotator.clear_currently_selected_segmentations()
        self.remove_layer_of_candidate_segmentation()
        self.set_segmentation_layer_visible()

    def modify_segmentation_and_tracklets_on_disk(self):
        # Uses segmentation as modified previously by candidate mask layer AND tracklet dataframe
        self.dat.segmentation_metadata.overwrite_original_detection_file()
        self.dat.tracklet_annotator.save_manual_matches_to_disk_dispatch()
        self.dat.modify_segmentation_on_disk_using_buffer()
        logging.info("Successfully saved to disk!")

    def split_segmentation_manual(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.split_current_neuron_and_add_napari_layer(self.viewer, split_method="Manual")
        self.set_segmentation_layer_invisible()

    def split_segmentation_automatic(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.split_current_neuron_and_add_napari_layer(self.viewer, split_method="Gaussian")
        self.set_segmentation_layer_invisible()

    def merge_segmentation(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.merge_current_neurons(self.viewer)
        self.set_segmentation_layer_invisible()

    def clear_current_segmentations(self):
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.clear_currently_selected_segmentations()

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="final_track", n_dimensional=True, symbol='cross', **points_opt,
                               visible=False)
        self.viewer.add_tracks(track_layer_data, name="track_of_point", visible=False)
        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)

        layer_to_add_callback = self.seg_layer
        added_segmentation_callbacks = [
            self.update_segmentation_status_label,
            self.toggle_highlight_selected_neuron
        ]
        added_tracklet_callbacks = [
            self.tracklet_updated_psuedo_event,
            self.set_segmentation_layer_invisible
        ]
        self.dat.tracklet_annotator.connect_tracklet_clicking_callback(
            layer_to_add_callback,
            self.viewer,
            added_segmentation_callbacks=added_segmentation_callbacks,
            added_tracklet_callbacks=added_tracklet_callbacks
        )
        self.update_neuron_in_tracklet_annotator()

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('r', overwrite=True)
        def refresh_subplot(viewer):
            self.update_trace_or_tracklet_subplot()

        @viewer.bind_key('d', overwrite=True)
        def zoom_next(viewer):
            self.zoom_next()

        @viewer.bind_key('a', overwrite=True)
        def zoom_previous(viewer):
            self.zoom_previous()

        @viewer.bind_key('f', overwrite=True)
        def zoom_to_next_nan(viewer):
            self.zoom_to_next_nan()

        @viewer.bind_key('g', overwrite=True)
        def zoom_to_next_nan(viewer):
            self.zoom_to_next_conflict()

        @viewer.bind_key('j', overwrite=True)
        def zoom_to_tracklet_end(viewer):
            self.zoom_to_end_of_current_tracklet()

        @viewer.bind_key('h', overwrite=True)
        def zoom_to_tracklet_end(viewer):
            self.zoom_to_start_of_current_tracklet()

        @viewer.bind_key('e', overwrite=True)
        def split_current_tracklet(viewer):
            self.split_current_tracklet_keep_right()

        @viewer.bind_key('q', overwrite=True)
        def split_current_tracklet(viewer):
            self.split_current_tracklet_keep_left()

        @viewer.bind_key('w', overwrite=True)
        def clear_current_tracklet(viewer):
            self.clear_current_tracklet()

        @viewer.bind_key('s', overwrite=True)
        def toggle_seg(viewer):
            # self.save_annotations_to_disk()
            self.toggle_raw_segmentation_layer()

        @viewer.bind_key('c', overwrite=True)
        def refresh_subplot(viewer):
            self.save_current_tracklet_to_neuron()

        @viewer.bind_key('v', overwrite=True)
        def print_tracklet_status(viewer):
            self.print_tracklets()

        # @viewer.bind_key('z', overwrite=True)
        # def remove_conflict(viewer):
        #     pass
            # self.remove_time_conflicts()

        @viewer.bind_key('x', overwrite=True)
        def remove_tracklet(viewer):
            self.save_segmentation_to_tracklet()
            # self.remove_tracklet_from_all_matches()

    @property
    def max_time(self):
        return len(self.dat.final_tracks) - 1

    def zoom_next(self, viewer=None):
        change_viewer_time_point(self.viewer, dt=1, a_max=self.max_time)
        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
        self.time_changed_callbacks()

    def zoom_previous(self, viewer=None):
        change_viewer_time_point(self.viewer, dt=-1, a_max=self.max_time)
        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
        self.time_changed_callbacks()

    def zoom_to_next_nan(self, viewer=None):
        y_on_plot = self.y_on_plot
        t = self.t
        if np.isnan(y_on_plot[t]):
            print("Already on nan point; not moving")
            return
        for i in range(t+1, len(y_on_plot)):
            if np.isnan(y_on_plot[i]):
                t_target = i
                change_viewer_time_point(self.viewer, t_target=t_target - 1)
                zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
                break
        else:
            print("No nan point found; not moving")
        self.time_changed_callbacks()

    def zoom_to_next_conflict(self, viewer=None):

        t, conflict_neuron = self.dat.tracklet_annotator.time_of_next_conflict(i_start=self.t)

        if conflict_neuron is not None:
            change_viewer_time_point(self.viewer, t_target=t)
            zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
        else:
            print("No conflict point found; not moving")
        self.time_changed_callbacks()

    def zoom_to_end_of_current_tracklet(self, viewer=None):
        t = self.dat.tracklet_annotator.end_time_of_current_tracklet()
        if t is not None:
            change_viewer_time_point(self.viewer, t_target=t)
            zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
        else:
            print("No tracklet selected; not zooming")
        self.time_changed_callbacks()

    def zoom_to_start_of_current_tracklet(self, viewer=None):
        t = self.dat.tracklet_annotator.start_time_of_current_tracklet()
        if t is not None:
            change_viewer_time_point(self.viewer, t_target=t)
            zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)
        else:
            print("No tracklet selected; not zooming")
        self.time_changed_callbacks()

    def split_current_tracklet_keep_right(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.remove_layer_of_current_tracklet()
            self.dat.tracklet_annotator.split_current_tracklet(self.t, True)
            self.add_layer_of_current_tracklet()
            self.tracklet_updated_psuedo_event()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def split_current_tracklet_keep_left(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.remove_layer_of_current_tracklet()
            self.dat.tracklet_annotator.split_current_tracklet(self.t + 1, False)
            self.add_layer_of_current_tracklet()
            self.tracklet_updated_psuedo_event()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def clear_current_tracklet(self):
        self.remove_layer_of_current_tracklet()
        self.dat.tracklet_annotator.clear_current_tracklet()
        self.tracklet_updated_psuedo_event()

    def toggle_raw_segmentation_layer(self):
        if self.viewer.layers.selection.active == self.seg_layer:
            self.viewer.layers.selection.clear()
            self.seg_layer.visible = False
        else:
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(self.seg_layer)
            self.seg_layer.visible = True

    def save_current_tracklet_to_neuron(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            tracklet_name = self.dat.tracklet_annotator.save_current_tracklet_to_current_neuron()
            if tracklet_name:
                self.remove_layer_of_current_tracklet(tracklet_name)
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")
        self.tracklet_updated_psuedo_event()

    def save_segmentation_to_tracklet(self):
        flag = self.dat.tracklet_annotator.attach_current_segmentation_to_current_tracklet()

    def remove_layer_of_current_tracklet(self, layer_name=None):
        if layer_name is None:
            layer_name = self.dat.tracklet_annotator.current_tracklet_name
        if layer_name is not None and layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

    def add_layer_of_current_tracklet(self, layer_name=None):
        if layer_name is None:
            layer_name = self.dat.tracklet_annotator.current_tracklet_name
        if layer_name is not None and layer_name not in self.viewer.layers:
            self.dat.tracklet_annotator.add_current_tracklet_to_viewer(self.viewer)

    def remove_layer_of_candidate_segmentation(self, layer_name='Candidate_mask'):
        if layer_name is not None and layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

    def print_tracklets(self):
        self.dat.tracklet_annotator.print_current_status()

    def remove_time_conflicts(self):
        self.dat.tracklet_annotator.remove_tracklets_with_time_conflicts()
        self.tracklet_updated_psuedo_event()

    def remove_tracklet_from_all_matches(self):
        self.dat.tracklet_annotator.remove_tracklet_from_all_matches()
        self.tracklet_updated_psuedo_event()

    @property
    def y_on_plot(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            y_list = self.y_tracklets
            if self.y_tracklet_current is not None:
                y_list.append(self.y_tracklet_current)
            tmp_df = y_list[0].copy()
            for df in y_list[1:]:
                tmp_df = tmp_df.combine_first(df)
            y_on_plot = tmp_df['z']
        else:
            y_on_plot = self.y_trace_mode
        return y_on_plot

    def init_universal_subplot(self):
        # Note: this causes a hang when the main window is closed, even though I'm trying to set the parent
        # ... workaround: ctrl-c necessary after closing
        # self.mpl_widget = PlotQWidget(self.viewer.window._qt_window.centralWidget())
        self.mpl_widget = PlotQWidget()
        self.static_ax = self.mpl_widget.canvas.fig.subplots()
        self.subplot_xlim = [0, self.dat.num_frames]
        # self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        # self.static_ax = self.mpl_widget.figure.subplots()
        # Connect clicking to a time change
        # https://matplotlib.org/stable/users/event_handling.html
        on_click = lambda event: self.on_subplot_click(event)
        cid = self.mpl_widget.canvas.mpl_connect('button_press_event', on_click)
        self.connect_time_line_callback()

    def init_subplot_post_clear(self):
        self.time_line = None
        self.time_line = self.static_ax.plot(*self.calculate_time_line())[0]
        if self.changeTraceTrackletDropdown.currentText() == "traces":
            self.static_ax.set_ylabel(self.changeTraceCalculationDropdown.currentText())
        else:
            self.static_ax.set_ylabel("z")
        self.color_using_behavior()
        self.static_ax.set_xlim(self.subplot_xlim)
        self.subplot_is_initialized = True

    def initialize_trace_subplot(self):
        self.update_stored_time_series()
        self.trace_line = self.static_ax.plot(self.tspan, self.y_trace_mode)[0]

    def initialize_tracklet_subplot(self):
        # Designed for traces, but reuse and force z coordinate
        self.update_stored_time_series('z')
        self.tracklet_lines = []
        self.update_stored_tracklets_for_plotting()
        for y in self.y_tracklets:
            this_line = self.static_ax.plot(self.tspan[:-1], y['z'])[0]
            self.tracklet_lines.append(this_line)
            # self.tracklet_lines.append(y['z'].plot(ax=self.static_ax))
        self.update_neuron_in_tracklet_annotator()
        # self.trace_line = self.static_ax.plot(self.y)[0]

    def on_subplot_click(self, event):
        t = event.xdata
        change_viewer_time_point(self.viewer, t_target=t)
        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)

    def change_trace_tracklet_mode(self):
        print(f"Changed mode to: {self.changeTraceTrackletDropdown.currentText()}")
        self.static_ax.clear()
        # Only show z coordinate for now
        self.changeChannelDropdown.setCurrentText('z')
        self.initialize_trace_or_tracklet_subplot()
        # Not just updating the data because we fully cleared the axes
        self.init_subplot_post_clear()

        self.finish_subplot_update(self.changeTraceTrackletDropdown.currentText())

    def initialize_trace_or_tracklet_subplot(self):
        if not self.subplot_is_initialized:
            self.init_universal_subplot()

        # This middle block will be called when the mode is switched
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            print("Initializing tracklet mode")
            self.initialize_tracklet_subplot()
        elif self.changeTraceTrackletDropdown.currentText() == 'traces':
            print("Initializing trace mode")
            self.initialize_trace_subplot()

        if not self.subplot_is_initialized:
            self.init_subplot_post_clear()
            # Finally, add the traces to napari
            self.viewer.window.add_dock_widget(self.mpl_widget, area='bottom')

    def update_trace_or_tracklet_subplot(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.update_tracklet_subplot()
        elif self.changeTraceTrackletDropdown.currentText() == 'traces':
            self.update_trace_subplot()
        else:
            raise ValueError

    def update_trace_subplot(self):
        if not self.changeTraceTrackletDropdown.currentText() == 'traces':
            print("Currently on tracklet setting, so this option didn't do anything")
            return
        self.update_stored_time_series()
        self.trace_line.set_ydata(self.y_trace_mode)
        title = f"{self.changeChannelDropdown.currentText()} trace for {self.changeTraceCalculationDropdown.currentText()} mode"

        time_options = self.calculate_time_line()
        self.time_line.set_data(time_options[:2])
        self.time_line.set_color(time_options[-1])
        self.finish_subplot_update(title)

    def update_tracklet_subplot(self):
        # For now, actually reinitializes the axes
        if not self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            print("Currently on traces setting, so this option didn't do anything")
            return

        # Tracklet unique part
        # if len(self.tracklet_lines) > 0:
        #     [t.remove() for t in self.tracklet_lines]
        # self.tracklet_lines = []
        self.update_stored_tracklets_for_plotting()
        self.static_ax.clear()
        for y in self.y_tracklets:
            self.tracklet_lines.append(y['z'].plot(ax=self.static_ax))
        if self.y_tracklet_current is not None:
            self.tracklet_lines.append(self.y_tracklet_current['z'].plot(ax=self.static_ax,
                                                                         color='k', lw=3))

        self.update_stored_time_series('z')  # Use this for the time line synchronization
        # We are displaying z here
        title = f"Tracklets for {self.changeNeuronsDropdown.currentText()}"

        self.init_subplot_post_clear()
        self.finish_subplot_update(title)

    def tracklet_updated_psuedo_event(self):
        self.update_tracklet_status_label()
        self.update_zoom_options_for_current_tracklet()
        self.add_to_recent_tracklet_dropdown()
        self.update_trace_or_tracklet_subplot()

    def update_tracklet_status_label(self):
        if self.dat.tracklet_annotator.current_neuron is None:
            update_string = "STATUS: No tracklet selected"
        else:
            if self.dat.tracklet_annotator.is_current_tracklet_confict_free:
                update_string = f"Selected: {self.dat.tracklet_annotator.current_tracklet_name}"
            else:
                types_of_conflicts = self.dat.tracklet_annotator.get_types_of_conflicts()
                update_string = f"Selected: {self.dat.tracklet_annotator.current_tracklet_name} " \
                                f"(Conflicts: {types_of_conflicts})"
        self.saveTrackletsStatusLabel.setText(update_string)

    def update_zoom_options_for_current_tracklet(self):
        tracklet_name = self.dat.tracklet_annotator.current_tracklet_name
        if tracklet_name and tracklet_name in self.viewer.layers:
            # Note that this should be called again if the layer is deleted
            self.zoom_opt['layer_name'] = self.dat.tracklet_annotator.current_tracklet_name
        else:
            # Set back to default
            self.zoom_opt['layer_name'] = 'final_track'

    def update_segmentation_status_label(self):
        if self.dat.tracklet_annotator.indices_of_original_neurons is None:
            update_string = "No segmentations selected"
        else:
            update_string = f"Selected segmentation(s): " \
                            f"{self.dat.tracklet_annotator.indices_of_original_neurons}"
        self.saveSegmentationStatusLabel.setText(update_string)

    def toggle_highlight_selected_neuron(self):
        self.dat.tracklet_annotator.toggle_highlight_selected_neuron(self.viewer)

    def center_on_selected_neuron(self):
        position = self.dat.tracklet_annotator.last_clicked_position
        # Only center if the last click was at the same time as the viewer
        if position is not None and position[0] == self.viewer.dims.current_step:
            zoom_using_viewer(position, self.viewer, zoom=None)

    def set_segmentation_layer_invisible(self):
        self.seg_layer.visible = False

    def set_segmentation_layer_visible(self):
        self.seg_layer.visible = True

    def set_segmentation_layer_do_not_show_selected_label(self):
        self.seg_layer.show_selected_label = False

    def time_changed_callbacks(self):
        self.set_segmentation_layer_do_not_show_selected_label()

    def finish_subplot_update(self, title):
        self.static_ax.set_title(title)
        self.color_using_behavior()
        self.static_ax.relim()
        self.static_ax.autoscale_view()
        self.mpl_widget.draw()

    def connect_time_line_callback(self):
        viewer = self.viewer

        @viewer.dims.events.current_step.connect
        def update_time_line(event):
            time_options = self.calculate_time_line()
            self.time_line.set_data(time_options[:2])
            self.time_line.set_color(time_options[-1])
            self.mpl_widget.draw()

    @property
    def t(self):
        return self.viewer.dims.current_step[0]

    def calculate_time_line(self):
        t = self.t
        y = self.y_on_plot
        ymin, ymax = np.min(y), np.max(y)
        if t <= len(y):
            self.tracking_is_nan = np.isnan(y[t])
        else:
            self.tracking_is_nan = True
        if self.tracking_is_nan:
            line_color = 'r'
        else:
            line_color = 'k'
        # print(f"Updating time line for t={t}, y[t] = {y[t]}, color={line_color}")
        return [t, t], [ymin, ymax], line_color

    def update_stored_time_series(self, calc_mode=None):
        # i = self.changeNeuronsDropdown.currentIndex()
        name = self.current_name
        channel = self.changeChannelDropdown.currentText()
        if calc_mode is None:
            calc_mode = self.changeTraceCalculationDropdown.currentText()
        remove_outliers_activity = False
        min_confidence = None
        # remove_outliers_activity = self.changeTraceOutlierCheckBox.checkState()
        # remove_outliers_tracking = self.changeTrackingOutlierCheckBox.checkState()
        # if remove_outliers_tracking:
        #     min_confidence = self.changeTrackingOutlierSpinBox.value()
        # else:
        #     min_confidence = None
        filter_mode = self.changeTraceFilteringDropdown.currentText()

        t, y = self.dat.calculate_traces(channel, calc_mode, name,
                                        remove_outliers_activity,
                                        filter_mode,
                                        min_confidence=min_confidence)
        self.y_trace_mode = y
        self.tspan = t

    def update_stored_tracklets_for_plotting(self):
        name = self.current_name
        tracklets, tracklet_current = self.dat.calculate_tracklets(name)
        print(f"Found {len(tracklets)} tracklets for {name}")
        if tracklet_current is not None:
            print("Additionally found 1 currently selected tracklet")
        self.y_tracklets = tracklets
        self.y_tracklet_current = tracklet_current

    def get_track_data(self):
        self.current_name = self.changeNeuronsDropdown.currentText()
        return self.build_tracks_from_name()

    def color_using_behavior(self):
        self.dat.shade_axis_using_behavior(self.static_ax)

    # def save_annotations(self):
    #     self.update_dataframe_using_points()
    #     # self.df[self.current_name] = new_df[self.current_name]
    #
    #     out_fname = self.annotation_output_name
    #     self.df.to_hdf(out_fname, 'df_with_missing')
    #
    #     out_fname = str(Path(out_fname).with_suffix('.csv'))
    #     #     df_old = pd.read_csv(out_fname)
    #     #     df_old[name] = df_new[name]
    #     #     df_old.to_csv(out_fname, mode='a')
    #     self.df.to_csv(out_fname)  # Just overwrite
    #
    #     print(f"Saved manual annotations for neuron {self.current_name} at {out_fname}")

    def update_dataframe_using_points(self):
        # Note: this allows for manual changing of the points
        new_df = self.build_df_of_current_points()

        self.dat.final_tracks = self.dat.final_tracks.drop(columns=self.current_name, level=0)
        self.dat.final_tracks = pd.concat([self.dat.final_tracks, new_df], axis=1)

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_name
        new_points = self.viewer.layers['final_track'].data

        # Initialize as dict and immediately create dataframe
        coords = ['z', 'x', 'y', 'likelihood']
        coords2ind = {'z': 1, 'x': 2, 'y': 3, 'likelihood': None}
        tmp_dict = {}
        for c in coords:
            key = (name, c)
            pts_ind = coords2ind[c]
            if pts_ind is not None:
                tmp_dict[key] = new_points[:, pts_ind]
            else:
                tmp_dict[key] = np.ones(new_points.shape[0])

        df_new = pd.DataFrame(tmp_dict)

        # col = pd.MultiIndex.from_product([[self.current_name], ['z', 'x', 'y', 'likelihood']])
        # df_new = pd.DataFrame(columns=col, index=self.dat.final_tracks.index)
        #
        # df_new[(name, 'z')] = new_points[:, 1]
        # df_new[(name, 'x')] = new_points[:, 2]
        # df_new[(name, 'y')] = new_points[:, 3]
        # df_new[(name, 'likelihood')] = np.ones(new_points.shape[0])

        return df_new

    def build_tracks_from_name(self):
        neuron_name = self.current_name
        df_single_track = self.dat.final_tracks[neuron_name]
        likelihood_threshold = self.dat.likelihood_thresh
        z_to_xy_ratio = self.dat.physical_unit_conversion.z_to_xy_ratio
        all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track,
                                                                                  likelihood_threshold,
                                                                                  z_to_xy_ratio)

        self.bad_points = to_remove
        return all_tracks_array, track_of_point


def napari_trace_explorer_from_config(project_path: str, to_print_fps=True, app=None):
    # A parent QT application must be initialized first
    os.environ["NAPARI_ASYNC"] = "1"
    if app is None:
        started_new_app = True
        app = QApplication([])
    else:
        started_new_app = False

    # Build object that has all the data
    project_data = ProjectData.load_final_project_data_from_config(project_path,
                                                                   to_load_tracklets=True,
                                                                   to_load_segmentation_metadata=True)
    ui, viewer = napari_trace_explorer(project_data, app=app, to_print_fps=to_print_fps)

    # Note: don't use this in jupyter
    napari.run()
    if started_new_app:
        app.exec_()
    logger.info("Quitting")
    sys.exit()


def napari_trace_explorer(project_data: ProjectData,
                          app: QApplication = None,
                          viewer: napari.Viewer = None,
                          to_print_fps: bool = False):
    """Current function for building the explorer (1/11/2022)"""
    print("Starting GUI setup")
    # Make sure ctrl-c works
    # https://python.tutorialink.com/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-console-ctrl-c/
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Build Napari and add data layers
    ui = NapariTraceExplorer(project_data, app)
    if viewer is None:
        logger.info("Creating a new Napari window")
        viewer = napari.Viewer(ndisplay=3)
    ui.dat.add_layers_to_viewer(viewer)

    # Actually dock my additional gui elements
    ui.setupUi(viewer)
    viewer.window.add_dock_widget(ui)
    ui.show()
    change_viewer_time_point(viewer, t_target=0)

    print("Finished GUI setup. If nothing is showing, trying quitting and running again")
    if to_print_fps:
        # From: https://github.com/napari/napari/issues/836
        def fps_status(viewer, x):
            # viewer.help = f'{x:.1f} frames per second'
            print(f'{x:.1f} frames per second')

        viewer.window.qt_viewer.canvas.measure_fps(callback=lambda x: fps_status(viewer, x))

    return ui, viewer
