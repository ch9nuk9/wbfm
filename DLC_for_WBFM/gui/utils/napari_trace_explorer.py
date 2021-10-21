import napari
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.visualization.filtering_traces import remove_outliers_via_rolling_mean, filter_rolling_mean, \
    filter_linear_interpolation
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from DLC_for_WBFM.gui.utils.utils_gui import zoom_using_viewer, change_viewer_time_point
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_labels_from_traces_dataframe
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


class NapariTraceExplorer(QtWidgets.QWidget):

    def __init__(self, project_path):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dat = ProjectData.load_final_project_data_from_config(project_path)

    def setupUi(self, viewer: napari.Viewer):

        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = list(self.dat.red_traces.columns.levels[0])
        self.current_name = neuron_names[0]

        # Change neurons (dropdown)
        self.groupBox1 = QtWidgets.QGroupBox("Neuron selection", self.verticalLayoutWidget)
        self.vbox1 = QtWidgets.QVBoxLayout(self.groupBox1)
        self.changeNeuronsDropdown = QtWidgets.QComboBox()
        self.changeNeuronsDropdown.addItems(neuron_names)
        self.changeNeuronsDropdown.setItemText(0, self.current_name)
        self.changeNeuronsDropdown.currentIndexChanged.connect(self.change_neurons)
        # self.verticalLayout.addWidget(self.changeNeuronsDropdown)
        self.vbox1.addWidget(self.changeNeuronsDropdown)

        # Change traces (dropdown)
        self.groupBox2 = QtWidgets.QGroupBox("Channel selection", self.verticalLayoutWidget)
        self.vbox2 = QtWidgets.QVBoxLayout(self.groupBox2)
        self.changeChannelDropdown = QtWidgets.QComboBox()
        self.changeChannelDropdown.addItems(['green', 'red', 'ratio'])
        self.changeChannelDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox2.addWidget(self.changeChannelDropdown)

        # Change traces (dropdown)
        self.groupBox3 = QtWidgets.QGroupBox("Trace calculation options", self.verticalLayoutWidget)
        self.vbox3 = QtWidgets.QVBoxLayout(self.groupBox3)
        self.changeTraceCalculationDropdown = QtWidgets.QComboBox()
        self.changeTraceCalculationDropdown.addItems(['integration', 'max', 'mean', 'z', 'volume'])
        self.changeTraceCalculationDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceCalculationDropdown)

        # Change trace filtering (dropdown)
        self.changeTraceFilteringDropdown = QtWidgets.QComboBox()
        self.changeTraceFilteringDropdown.addItems(['no_filtering', 'rolling_mean', 'linear_interpolation'])
        self.changeTraceFilteringDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceFilteringDropdown)

        # Change trace outlier removal (dropdown)
        self.changeTraceOutlierCheckBox = QtWidgets.QCheckBox("Remove outliers?")
        # self.changeTraceOutlierDropdown.addItems(['no_filtering', 'rolling_mean', 'linear_interpolation'])
        self.changeTraceOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        self.vbox3.addWidget(self.changeTraceOutlierCheckBox)

        self.verticalLayout.addWidget(self.groupBox1)
        self.verticalLayout.addWidget(self.groupBox2)
        self.verticalLayout.addWidget(self.groupBox3)

        # Save annotations (button)
        # self.saveButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        # self.saveButton.clicked.connect(self.save_annotations)
        # self.saveButton.setText("Save Annotations")
        # self.verticalLayout.addWidget(self.saveButton)

        self.initialize_track_layers()
        self.initialize_shortcuts()
        self.initialize_trace_subplot()

    def change_neurons(self):
        self.update_dataframe_using_points()
        self.update_track_layers()
        self.update_trace_subplot()

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['final_track'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_viewer(self.viewer, layer_name='final_track')

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="final_track", n_dimensional=True, symbol='cross', **points_opt)

        self.viewer.add_tracks(track_layer_data, name="track_of_point")

        zoom_using_viewer(self.viewer, layer_name='final_track', zoom=10)

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('.', overwrite=True)
        def zoom_next(viewer):
            change_viewer_time_point(viewer, dt=1, a_max=len(self.dat.final_tracks) - 1)
            zoom_using_viewer(viewer, layer_name='final_track', zoom=None)

        @viewer.bind_key(',', overwrite=True)
        def zoom_previous(viewer):
            change_viewer_time_point(viewer, dt=-1, a_max=len(self.dat.final_tracks) - 1)
            zoom_using_viewer(viewer, layer_name='final_track', zoom=None)

    def initialize_trace_subplot(self):
        self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.static_ax = self.mpl_widget.figure.subplots()
        self.update_stored_time_series()
        self.trace_line = self.static_ax.plot(self.y)[0]
        self.time_line = self.static_ax.plot(*self.calculate_time_line())[0]
        self.color_using_behavior()
        self.connect_time_line_callback()

        # Connect clicking to a time change
        # https://matplotlib.org/stable/users/event_handling.html
        on_click = lambda event: self.on_trace_plot_click(event)
        cid = self.mpl_widget.mpl_connect('button_press_event', on_click)

        # Finally, add the traces to napari
        self.viewer.window.add_dock_widget(self.mpl_widget, area='bottom')

    def on_trace_plot_click(self, event):
        t = event.xdata
        change_viewer_time_point(self.viewer, t_target=t)

    def update_trace_subplot(self):
        self.update_stored_time_series()
        self.trace_line.set_ydata(self.y)
        self.time_line.set_data(self.calculate_time_line()[:2])
        self.color_using_behavior()
        title = f"{self.changeChannelDropdown.currentText()} trace for {self.changeTraceCalculationDropdown.currentText()} mode"
        self.static_ax.set_title(title)

        self.static_ax.relim()
        self.static_ax.autoscale_view()
        self.mpl_widget.draw()

    def connect_time_line_callback(self):
        viewer = self.viewer

        @viewer.dims.events.current_step.connect
        def update_time_line(event):
            self.time_line.set_data(self.calculate_time_line()[:2])
            self.mpl_widget.draw()

    def calculate_time_line(self):
        t = self.viewer.dims.current_step[0]
        y = self.y
        ymin, ymax = np.min(y), np.max(y)
        self.tracking_lost = not np.isnan(y[t])
        if not self.tracking_lost:
            # z, x, y = self.current_centroid
            # title = f"{current_neuron}: {mode} trace at ({z:.1f}, {x:.0f}, {y:.0f})"
            line_color = 'k'
        else:
            # title = "Tracking lost!"
            line_color = 'k'
        return [t, t], [ymin, ymax], line_color

    def update_stored_time_series(self):
        # i = self.changeNeuronsDropdown.currentIndex()
        name = self.current_name
        channel = self.changeChannelDropdown.currentText()
        calc_mode = self.changeTraceCalculationDropdown.currentText()
        remove_outliers = self.changeTraceOutlierCheckBox.checkState()
        filter_mode = self.changeTraceFilteringDropdown.currentText()

        y = self.dat.calculate_traces(channel, calc_mode, name, remove_outliers, filter_mode)

        self.y = y

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
        # Just visualize one neuron for now
        # 5 columns:
        # track_id, t, z, y, x
        likelihood_thresh = self.dat.likelihood_thresh
        try:
            coords = ['z_dlc', 'x_dlc', 'y_dlc']
            zxy_array = np.array(self.dat.red_traces[self.current_name][coords])
        except KeyError:
            coords = ['z', 'x', 'y']
            zxy_array = np.array(self.dat.red_traces[self.current_name][coords])

        all_tracks_list = []
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

        if likelihood_thresh is not None and 'likelihood' in self.dat.final_tracks[self.current_name]:
            to_remove = self.dat.final_tracks[self.current_name]['likelihood'] < likelihood_thresh
        else:
            to_remove = np.zeros_like(zxy_array[:, 0], dtype=bool)
        zxy_array[to_remove, :] = 0

        all_tracks_list.append(np.hstack([t_array, zxy_array]))
        all_tracks_array = np.vstack(all_tracks_list)

        track_of_point = np.hstack([np.ones((all_tracks_array.shape[0], 1)), all_tracks_array])

        self.bad_points = to_remove
        return all_tracks_array, track_of_point


def build_napari_trace_explorer(project_config):
    viewer = napari.Viewer(ndisplay=2)

    # Build object that has all the data
    ui = NapariTraceExplorer(project_config)

    # Build Napari and add widgets
    print("Finished loading data, starting napari...")
    viewer.add_image(ui.dat.red_data, name="Red data", opacity=0.5, colormap='red', visible=False)
    viewer.add_image(ui.dat.green_data, name="Green data", opacity=0.5, colormap='green')
    viewer.add_labels(ui.dat.raw_segmentation, name="Raw segmentation", opacity=0.4, visible=False)
    if ui.dat.segmentation is not None:
        viewer.add_labels(ui.dat.segmentation, name="Colored segmentation", opacity=0.4)

    # Add a text overlay
    df = ui.dat.red_traces
    options = napari_labels_from_traces_dataframe(df)
    viewer.add_points(**options)

    # Actually dock my additional gui elements
    ui.setupUi(viewer)
    viewer.window.add_dock_widget(ui)
    ui.show()

    print("Finished GUI setup")

    napari.run()
