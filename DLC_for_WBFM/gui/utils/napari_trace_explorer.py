import argparse
import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from DLC_for_WBFM.gui.utils.utils_gui import zoom_using_viewer, change_viewer_time_point
from DLC_for_WBFM.utils.projects.utils_project import safe_cd, load_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import get_or_recalculate_which_frames


class napari_trace_explorer(QtWidgets.QWidget):

    def __init__(self, project_config):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)

        ########################
        # Load Configs
        ########################
        cfg, traces_cfg, tracking_cfg = self._load_config_files(project_config)
        self._load_data(cfg, traces_cfg, tracking_cfg)

        # Temporary
        self.df = self.dlc_raw

    def _load_data(self, cfg, traces_cfg, tracking_cfg):
        ########################
        # Load Data
        ########################
        # Raw data
        red_dat_fname = cfg['preprocessed_red']
        green_dat_fname = cfg['preprocessed_green']
        self.red_data = zarr.open(red_dat_fname)
        self.green_data = zarr.open(green_dat_fname)

        with safe_cd(self.project_dir):
            # Traces
            red_traces_fname = traces_cfg['traces']['red']
            green_traces_fname = traces_cfg['traces']['green']
            dlc_raw_fname = tracking_cfg['final_3d_tracks']['df_fname']
            self.red_traces = pd.read_hdf(red_traces_fname)
            self.green_traces = pd.read_hdf(green_traces_fname)
            self.dlc_raw = pd.read_hdf(dlc_raw_fname)

            # Segmentation
            seg_fname_raw = self.segment_cfg['output']['masks']
            if '.zarr' in seg_fname_raw:
                self.raw_segmentation = zarr.open(seg_fname_raw, mode='r')
            else:
                self.raw_segmentation = None

            seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')
            if os.path.exists(seg_fname):
                self.segmentation = zarr.open(seg_fname, mode='r')
            else:
                self.segmentation = None

        # TODO: do not hardcode
        self.background_per_pixel = 15


    def _load_config_files(self, project_config):
        self.project_dir = Path(project_config).parent
        cfg = load_config(project_config)
        self.cfg = cfg
        with safe_cd(self.project_dir):
            traces_cfg = load_config(cfg['subfolder_configs']['traces'])
            self.traces_cfg = traces_cfg
            segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
            self.segment_cfg = segment_cfg
            tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])
            self.tracking_cfg = tracking_cfg
        # self.crop_sz = (1, 48, 48)
        start = cfg['dataset_params']['start_volume']
        end = start + cfg['dataset_params']['num_frames']
        self.x = list(range(start, end))
        return cfg, traces_cfg, tracking_cfg

    def setupUi(self, viewer: napari.Viewer):

        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = list(self.dlc_raw.columns.levels[0])
        self.current_name = neuron_names[0]

        # Change neurons (dropdown)
        self.changeNeuronsDropdown = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.changeNeuronsDropdown.addItems(neuron_names)
        self.changeNeuronsDropdown.setItemText(0, self.current_name)
        self.changeNeuronsDropdown.currentIndexChanged.connect(self.change_neurons)
        self.verticalLayout.addWidget(self.changeNeuronsDropdown)

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

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['deeplabcut_track'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_viewer(self.viewer, layer_name='deeplabcut_track')

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="deeplabcut_track", n_dimensional=True, symbol='cross', **points_opt)

        self.viewer.add_tracks(track_layer_data, name="track_of_point")

        zoom_using_viewer(self.viewer, layer_name='deeplabcut_track', zoom=10)

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('.', overwrite=True)
        def zoom_next(viewer):
            change_viewer_time_point(viewer, dt=1, a_max=len(self.df) - 1)
            zoom_using_viewer(viewer, layer_name='deeplabcut_track', zoom=None)

        @viewer.bind_key(',', overwrite=True)
        def zoom_previous(viewer):
            change_viewer_time_point(viewer, dt=-1, a_max=len(self.df) - 1)
            zoom_using_viewer(viewer, layer_name='deeplabcut_track', zoom=None)

    def initialize_trace_subplot(self):
        self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.static_ax = self.mpl_widget.figure.subplots()
        self.trace_line = self.static_ax.plot(self.calculate_trace())
        # t = np.linspace(0, 10, 501)
        # static_ax.plot(t, np.tan(t), ".")

        # self.mpl_widget.draw()
        self.viewer.window.add_dock_widget(self.mpl_widget, area='bottom')

    def update_trace_subplot(self):
        self.trace_line.set_ydata(self.calculate_trace())
        self.mpl_widget.draw()


    def calculate_trace(self):
        # i = self.changeNeuronsDropdown.currentIndex()
        i = self.current_name

        df = self.red_traces
        # print(df)
        y_raw = df[i]['brightness']
        smoothing_func = lambda x: x
        y = smoothing_func(y_raw - self.background_per_pixel * df[i]['volume'])
        self.y = y
        return y

    def get_track_data(self):
        self.current_name = self.changeNeuronsDropdown.currentText()
        return self.build_tracks_from_name()

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

        self.df = self.df.drop(columns=self.current_name, level=0)
        self.df = pd.concat([self.df, new_df], axis=1)

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_name
        new_points = self.viewer.layers['deeplabcut_track'].data

        col = pd.MultiIndex.from_product([[self.current_name], ['z', 'x', 'y', 'likelihood']])
        df_new = pd.DataFrame(columns=col, index=self.df.index)

        # df_new[(name, 't')] = new_points[:, 0]
        df_new[(name, 'z')] = new_points[:, 1]
        df_new[(name, 'y')] = new_points[:, 2]
        df_new[(name, 'x')] = new_points[:, 3]
        df_new[(name, 'likelihood')] = np.ones(new_points.shape[0])

        return df_new

    def build_tracks_from_name(self):
        # Just visualize one neuron for now
        # 5 columns:
        # track_id, t, z, y, x
        coords = ['z', 'y', 'x']
        all_tracks_list = []
        likelihood_thresh = 0.4
        zxy_array = np.array(self.df[self.current_name][coords])
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)
        # Remove low likelihood
        to_remove = self.df[self.current_name]['likelihood'] < likelihood_thresh
        zxy_array[to_remove, :] = 0

        all_tracks_list.append(np.hstack([t_array, zxy_array]))
        all_tracks_array = np.vstack(all_tracks_list)

        track_of_point = np.hstack([np.ones((all_tracks_array.shape[0], 1)), all_tracks_array])

        self.bad_points = to_remove
        return all_tracks_array, track_of_point


def build_napari_trace_explorer(project_config):

    viewer = napari.Viewer(ndisplay=2)

    # Build object that has all the data
    ui = napari_trace_explorer(project_config)

    # Build Napari and add widgets
    print("Finished loading data, starting napari...")
    viewer.add_image(ui.red_data, name="Red data", opacity=0.5)
    viewer.add_labels(ui.raw_segmentation, name="Raw segmentation", opacity=0.5)
    if ui.segmentation is not None:
        viewer.add_labels(ui.segmentation)

    # Actually dock my additional gui elements
    ui.setupUi(viewer)
    viewer.window.add_dock_widget(ui)
    ui.show()

    print("Finished GUI setup")

    napari.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    DEBUG = args.DEBUG

    print("Starting trace explorer GUI, may take a while to load...")

    build_napari_trace_explorer(project_path)