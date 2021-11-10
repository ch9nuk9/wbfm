import logging
import napari
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from DLC_for_WBFM.gui.utils.utils_gui import zoom_using_viewer, change_viewer_time_point, build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


class NapariTraceExplorer(QtWidgets.QWidget):

    subplot_is_initialized = False
    tracklet_lines = []

    def __init__(self, project_data: ProjectData):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dat = project_data

    def setupUi(self, viewer: napari.Viewer):

        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = list(self.dat.red_traces.columns.levels[0])
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
        self.groupBox2 = QtWidgets.QGroupBox("Channel selection", self.verticalLayoutWidget)
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

        # BOX 3: Trace filtering / display options
        # Change traces (dropdown)
        self.groupBox3 = QtWidgets.QGroupBox("Trace calculation options", self.verticalLayoutWidget)
        self.vbox3 = QtWidgets.QVBoxLayout(self.groupBox3)
        self.changeTraceCalculationDropdown = QtWidgets.QComboBox()
        self.changeTraceCalculationDropdown.addItems(['integration', 'z', 'volume', 'likelihood'])
        self.changeTraceCalculationDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceCalculationDropdown)

        # Change trace filtering (checkbox)
        self.changeTraceFilteringDropdown = QtWidgets.QComboBox()
        self.changeTraceFilteringDropdown.addItems(['no_filtering', 'rolling_mean', 'linear_interpolation'])
        self.changeTraceFilteringDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceFilteringDropdown)

        # Change trace outlier removal (dropdown)
        self.changeTraceOutlierCheckBox = QtWidgets.QCheckBox("Remove outliers (activity)?")
        self.changeTraceOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceOutlierCheckBox)

        self.changeTrackingOutlierCheckBox = QtWidgets.QCheckBox("Remove outliers (tracking confidence)?")
        self.changeTrackingOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTrackingOutlierCheckBox)

        self.changeTrackingOutlierSpinBox = QtWidgets.QSpinBox()
        self.changeTrackingOutlierSpinBox.setRange(0, 1)
        self.changeTrackingOutlierSpinBox.setSingleStep(0.1)
        self.changeTrackingOutlierSpinBox.valueChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTrackingOutlierSpinBox)

        # BOX 4: general shortcuts
        self.groupBox4 = QtWidgets.QGroupBox("Shortcuts", self.verticalLayoutWidget)
        self.vbox4 = QtWidgets.QVBoxLayout(self.groupBox4)

        self.refreshButton = QtWidgets.QPushButton("Refresh Subplot (R)")
        self.refreshButton.pressed.connect(self.update_trace_or_tracklet_subplot)
        self.vbox4.addWidget(self.refreshButton)

        self.zoom1Button = QtWidgets.QPushButton("Zoom next (D)")
        self.zoom1Button.pressed.connect(self.zoom_next)
        self.vbox4.addWidget(self.zoom1Button)
        self.zoom2Button = QtWidgets.QPushButton("Zoom previous (A)")
        self.zoom2Button.pressed.connect(self.zoom_previous)
        self.vbox4.addWidget(self.zoom2Button)
        self.zoom3Button = QtWidgets.QPushButton("Zoom to next nan (F)")
        self.zoom3Button.pressed.connect(self.zoom_to_next_nan)
        self.vbox4.addWidget(self.zoom3Button)

        self.splitTrackletButton = QtWidgets.QPushButton("Split current tracklet (E)")
        self.splitTrackletButton.pressed.connect(self.split_current_tracklet)
        self.vbox4.addWidget(self.splitTrackletButton)

        self.appendTrackletButton = QtWidgets.QPushButton("Append current tracklet to neuron (Q)")
        self.appendTrackletButton.pressed.connect(self.append_current_tracklet_to_dict)
        self.vbox4.addWidget(self.appendTrackletButton)

        self.saveTrackletsButton = QtWidgets.QPushButton("Save manual annotations to disk (S)")
        self.saveTrackletsButton.pressed.connect(self.save_annotations_to_disk)
        self.vbox4.addWidget(self.saveTrackletsButton)

        self.printTrackletsButton = QtWidgets.QPushButton("Print current tracklets (V)")
        self.printTrackletsButton.pressed.connect(self.print_tracklets)
        self.vbox4.addWidget(self.printTrackletsButton)

        self.verticalLayout.addWidget(self.groupBox1)
        self.verticalLayout.addWidget(self.groupBox2)
        self.verticalLayout.addWidget(self.groupBox3)
        self.verticalLayout.addWidget(self.groupBox4)

        # General
        # self.verticalLayout.addWidget(self.refreshButton)

        # Save annotations (button)
        # self.saveButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        # self.saveButton.clicked.connect(self.save_annotations)
        # self.saveButton.setText("Save Annotations")
        # self.verticalLayout.addWidget(self.saveButton)

        self.initialize_track_layers()
        self.initialize_shortcuts()
        self.initialize_trace_or_tracklet_subplot()

    def change_neurons(self):
        self.update_dataframe_using_points()
        self.update_track_layers()
        self.update_trace_or_tracklet_subplot()
        self.update_tracklet_annotator()

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['final_track'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_viewer(self.viewer, layer_name='final_track')

    def update_tracklet_annotator(self):
        self.dat.tracklet_annotator.current_neuron = self.changeNeuronsDropdown.currentText()

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="final_track", n_dimensional=True, symbol='cross', **points_opt)
        self.viewer.add_tracks(track_layer_data, name="track_of_point")
        zoom_using_viewer(self.viewer, layer_name='final_track', zoom=10)

        layer_to_add_callback = self.viewer.layers['Raw segmentation']
        self.dat.tracklet_annotator.connect_tracklet_clicking_callback(
            layer_to_add_callback,
            self.viewer,
            refresh_callback=self.update_trace_or_tracklet_subplot
        )
        self.update_tracklet_annotator()

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

        @viewer.bind_key('e', overwrite=True)
        def split_current_tracklet(viewer):
            self.split_current_tracklet()

        @viewer.bind_key('s', overwrite=True)
        def refresh_subplot(viewer):
            self.save_annotations_to_disk()

        @viewer.bind_key('q', overwrite=True)
        def refresh_subplot(viewer):
            self.append_current_tracklet_to_dict()

        @viewer.bind_key('v', overwrite=True)
        def print_tracklet_status(viewer):
            self.print_tracklets()

    @property
    def max_time(self):
        return len(self.dat.final_tracks) - 1

    def zoom_next(self, viewer=None):
        change_viewer_time_point(self.viewer, dt=1, a_max=self.max_time)
        zoom_using_viewer(self.viewer, layer_name='final_track', zoom=None)

    def zoom_previous(self, viewer=None):
        change_viewer_time_point(self.viewer, dt=-1, a_max=self.max_time)
        zoom_using_viewer(self.viewer, layer_name='final_track', zoom=None)

    def zoom_to_next_nan(self, viewer=None):
        y_on_plot = self.y_on_plot()
        t = self.t
        for i in range(t, len(y_on_plot)):
            if np.isnan(y_on_plot[i]):
                t_target = i
                change_viewer_time_point(self.viewer, t_target=t_target - 1)
                zoom_using_viewer(self.viewer, layer_name='final_track', zoom=None)
                break
        else:
            print("No nan point found; not switching")

    def split_current_tracklet(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.dat.tracklet_annotator.split_current_tracklet(self.t)
            self.update_trace_or_tracklet_subplot()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def save_annotations_to_disk(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.dat.tracklet_annotator.save_manual_matches_to_disk()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def append_current_tracklet_to_dict(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.dat.tracklet_annotator.save_current_tracklet_to_neuron()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def print_tracklets(self):
        self.dat.tracklet_annotator.print_current_status()

    def y_on_plot(self):
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            y_list = self.dat.tracklet_annotator.calculate_tracklets_for_neuron()
            tmp_df = y_list[0].copy()
            for df in y_list[1:]:
                tmp_df = tmp_df.combine_first(df)
            # y_on_plot = np.sum(y_list, axis=1)
            y_on_plot = tmp_df['z']
        else:
            y_on_plot = self.y
        return y_on_plot

    def init_universal_subplot(self):
        self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.static_ax = self.mpl_widget.figure.subplots()
        # Connect clicking to a time change
        # https://matplotlib.org/stable/users/event_handling.html
        on_click = lambda event: self.on_subplot_click(event)
        cid = self.mpl_widget.mpl_connect('button_press_event', on_click)

    def init_subplot_post_clear(self):
        self.time_line = self.static_ax.plot(*self.calculate_time_line())[0]
        self.static_ax.set_ylabel(self.changeTraceCalculationDropdown.currentText())
        self.color_using_behavior()
        self.connect_time_line_callback()
        self.subplot_is_initialized = True

    def initialize_trace_subplot(self):
        self.update_stored_time_series()
        self.trace_line = self.static_ax.plot(self.y)[0]

    def initialize_tracklet_subplot(self):
        # Designed for traces, but reuse and force z coordinate
        self.update_stored_time_series('z')
        self.tracklet_lines = []
        self.update_stored_tracklets()
        for y in self.y_tracklets:
            self.tracklet_lines.append(y['z'].plot(ax=self.static_ax))
        self.update_tracklet_annotator()
        # self.trace_line = self.static_ax.plot(self.y)[0]

    def on_subplot_click(self, event):
        t = event.xdata
        change_viewer_time_point(self.viewer, t_target=t)

    def change_trace_tracklet_mode(self):
        print(f"Changed mode to: {self.changeTraceTrackletDropdown.currentText()}")
        self.static_ax.clear()
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
        self.trace_line.set_ydata(self.y)
        title = f"{self.changeChannelDropdown.currentText()} trace for {self.changeTraceCalculationDropdown.currentText()} mode"

        time_options = self.calculate_time_line()
        self.time_line.set_data(time_options[:2])
        self.time_line.color = time_options[-1]
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
        self.update_stored_tracklets()
        self.static_ax.clear()
        for y in self.y_tracklets:
            self.tracklet_lines.append(y['z'].plot(ax=self.static_ax))

        self.update_stored_time_series('z')  # Use this for the time line synchronization
        # We are displaying z here
        title = f"Tracklets for neuron {self.changeNeuronsDropdown.currentText()}"

        self.init_subplot_post_clear()
        self.finish_subplot_update(title)

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
            self.time_line.color = time_options[-1]
            self.mpl_widget.draw()

    @property
    def t(self):
        return self.viewer.dims.current_step[0]

    def calculate_time_line(self):
        t = self.t
        y = self.y
        print(f"Calculating time line for t={t}")
        ymin, ymax = np.min(y), np.max(y)
        self.tracking_is_nan = np.isnan(y[t])
        print(f"Current point: {y[t]}")
        if self.tracking_is_nan:
            line_color = 'r'
        else:
            line_color = 'k'
        return [t, t], [ymin, ymax], line_color

    def update_stored_time_series(self, calc_mode=None):
        # i = self.changeNeuronsDropdown.currentIndex()
        name = self.current_name
        channel = self.changeChannelDropdown.currentText()
        if calc_mode is None:
            calc_mode = self.changeTraceCalculationDropdown.currentText()
        remove_outliers_activity = self.changeTraceOutlierCheckBox.checkState()
        remove_outliers_tracking = self.changeTrackingOutlierCheckBox.checkState()
        if remove_outliers_tracking:
            min_confidence = self.changeTrackingOutlierSpinBox.value()
        else:
            min_confidence = None
        filter_mode = self.changeTraceFilteringDropdown.currentText()

        y = self.dat.calculate_traces(channel, calc_mode, name,
                                      remove_outliers_activity,
                                      filter_mode,
                                      min_confidence=min_confidence)
        self.y = y

    def update_stored_tracklets(self):
        name = self.current_name
        tracklets = self.dat.calculate_tracklets(name)
        print(f"Found {len(tracklets)} tracklets for {name}")
        self.y_tracklets = tracklets

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
        all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track, likelihood_threshold)

        self.bad_points = to_remove
        return all_tracks_array, track_of_point


def napari_trace_explorer_from_config(project_path: str, to_print_fps=True):

    # Build object that has all the data
    project_data = ProjectData.load_final_project_data_from_config(project_path, to_load_tracklets=True)
    napari_trace_explorer(project_data, to_print_fps=to_print_fps)

    # Note: don't use this in jupyter
    napari.run()


def napari_trace_explorer(project_data: ProjectData,
                          viewer: napari.Viewer = None,
                          to_print_fps: bool = False):
    print("Starting GUI setup")
    ui = NapariTraceExplorer(project_data)
    # Build Napari and add widgets
    if viewer is None:
        viewer = napari.Viewer(ndisplay=2)
    ui.dat.add_layers_to_viewer(viewer)
    # Actually dock my additional gui elements
    ui.setupUi(viewer)
    viewer.window.add_dock_widget(ui)
    ui.show()
    print("Finished GUI setup")
    if to_print_fps:
        # From: https://github.com/napari/napari/issues/836
        def fps_status(viewer, x):
            # viewer.help = f'{x:.1f} frames per second'
            print(f'{x:.1f} frames per second')

        viewer.window.qt_viewer.canvas.measure_fps(callback=lambda x: fps_status(viewer, x))

    return ui, viewer
