import argparse
import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr
from PyQt5 import QtWidgets

from DLC_for_WBFM.utils.projects.utils_project import safe_cd, load_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import get_or_recalculate_which_frames


class manual_annotation_widget(QtWidgets.QWidget):

    def __init__(self):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)

    #     def setupUi(self, MainWindow, df, output_dir, viewer: napari.Viewer):
    def setupUi(self, df, output_dir, viewer: napari.Viewer):

        # Load dataframe and path to outputs
        self.viewer = viewer
        self.output_dir = output_dir
        self.df = df
        neuron_names = list(df.columns.levels[0])
        self.current_name = neuron_names[0]

        # Change neurons (dropdown)
        self.changeNeuronsButton = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.changeNeuronsButton.addItems(neuron_names)
        self.changeNeuronsButton.setItemText(0, self.current_name)
        self.changeNeuronsButton.currentIndexChanged.connect(self.change_neurons)
        self.verticalLayout.addWidget(self.changeNeuronsButton)

        # Change zoom (slider)
        self.changeZoomSlider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.changeZoomSlider.setMinimum(1)
        self.changeZoomSlider.setMaximum(20)
        self.changeZoomSlider.valueChanged.connect(self.zoom_and_change_time)
        self.verticalLayout.addWidget(self.changeZoomSlider)


        # Save annotations (button)
        self.saveButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.saveButton.clicked.connect(self.save_annotations)
        self.saveButton.setText("Save Annotations")
        self.verticalLayout.addWidget(self.saveButton)

        self.initialize_track_layers()
        self.initialize_shortcuts()

    def change_neurons(self):
        # Save current track (overwrite old df)
        new_df = self.build_df_of_current_points()
        self.df[self.current_name] = new_df[self.current_name]

        self.update_track_layers()

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['pts_with_future_and_past'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_viewer(self.viewer)

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="pts_with_future_and_past", n_dimensional=True, symbol='square', **points_opt)

        self.viewer.add_tracks(track_layer_data, name="track_of_point")

        zoom_using_viewer(self.viewer)

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('.', overwrite=True)
        def zoom_next():
            self.zoom_and_change_time(1)

        @viewer.bind_key(',', overwrite=True)
        def zoom_previous():
            self.zoom_and_change_time(-1)

    def zoom_and_change_time(self, dt=0):
        viewer = self.viewer
        zoom = self.get_zoom()
        change_viewer_time_point(viewer, dt)
        zoom_using_viewer(viewer, zoom)


    def get_zoom(self):
        return 10

    def get_track_data(self):
        self.current_name = self.changeNeuronsButton.currentText()
        return self.build_tracks_from_name()

    def save_annotations(self):
        out_fname = os.path.join(self.output_dir, f'corrected_tracks.h5')
        self.df.to_hdf(out_fname, 'df_with_missing')

        out_fname = str(Path(out_fname).with_suffix('.csv'))
        #     df_old = pd.read_csv(out_fname)
        #     df_old[name] = df_new[name]
        #     df_old.to_csv(out_fname, mode='a')
        self.df.to_csv(out_fname)  # Just overwrite

        print(f"Saved manual annotations for neuron {self.current_name} at {out_fname}")

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_name
        new_points = self.viewer.layers['pts_with_future_and_past'].data

        col = pd.MultiIndex.from_product([[self.current_name], ['t', 'z', 'x', 'y']])
        df_new = pd.DataFrame(columns=col)

        df_new[(name, 't')] = new_points[:, 0]
        df_new[(name, 'z')] = new_points[:, 1]
        df_new[(name, 'y')] = new_points[:, 2]
        df_new[(name, 'x')] = new_points[:, 3]
        df_new[(name, 'likelihood')] = self.df[(name, 'likelihood')]  # Same as before

        df_new.sort_values((name, 't'), inplace=True, ignore_index=True)

        # print(df_new)

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
        # zxy_array = zxy_array[to_keep, :]
        # t_array = t_array[to_keep, :]
        zxy_array[to_remove, :] = 0

        all_tracks_list.append(np.hstack([t_array, zxy_array]))
        all_tracks_array = np.vstack(all_tracks_list)

        track_of_point = np.hstack([np.ones((all_tracks_array.shape[0], 1)), all_tracks_array])

        self.bad_points = to_remove
        return all_tracks_array, track_of_point


def zoom_using_viewer(viewer: napari.Viewer, zoom=10) -> None:
    # Get current point
    t = viewer.dims.current_step[0]
    tzxy = viewer.layers['pts_with_future_and_past'].data[t]

    # Zoom to it in XY
    viewer.camera.zoom = zoom
    viewer.camera.center = tzxy[1:]

    # Zoom in Z
    if tzxy[2] > 0 and tzxy[3] > 0:
        viewer.dims.current_step = (t, tzxy[1], 0, 0)


def change_viewer_time_point(viewer: napari.Viewer, dt: int) -> None:
    # Increment time
    t = viewer.dims.current_step[0] + dt
    tzxy = (t,) + viewer.dims.current_step[1:]
    viewer.dims.current_step = tzxy


def create_manual_correction_gui(this_config, DEBUG=False):
    """
    Creates a napari-based gui for correcting tracks

    For now, only works with training data
    """
    project_dir = this_config['project_dir']

    with safe_cd(project_dir):

        fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
        df = pd.read_pickle(fname)

        # Get the frames chosen as training data, or recalculate
        which_frames = list(get_or_recalculate_which_frames(DEBUG, df, this_config))

        # Import segmentation
        fname = this_config['segment_cfg']['output']['masks']
        raw_segmentation = zarr.open(fname)

        fname = os.path.join('2-training_data', 'training_data_tracks.h5')
        df = pd.read_hdf(fname)

        # Import raw data
        fname = this_config['project_cfg']['preprocessed_red']
        red_data = zarr.open(fname)

    # Build Napari and add widgets
    viewer = napari.view_image(red_data[which_frames[0]:which_frames[-1]+1, ...], name="Red data", ndisplay=2, opacity=0.5)
    viewer.add_labels(raw_segmentation[which_frames[0]:which_frames[-1]+1, ...], name="Raw segmentation")

    output_dir = os.path.join("2-training_data", "manual_tracking")
    ui = manual_annotation_widget()
    # TODO: not hardcoded experimenter
    ui.setupUi(df['Charlie'], output_dir, viewer)

    # Actually dock
    viewer.window.add_dock_widget(ui)
    ui.show()

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

    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        trace_fname = Path(project_cfg['subfolder_configs']['traces'])
        trace_cfg = dict(load_config(trace_fname))
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))
        seg_fname = Path(project_cfg['subfolder_configs']['segmentation'])
        segment_cfg = dict(load_config(seg_fname))

    this_config = {'track_cfg': track_cfg, 'segment_cfg': segment_cfg, 'project_cfg': project_cfg,
                   'dataset_params': project_cfg['dataset_params'].copy(),
                   'project_dir': project_dir}

    with safe_cd(project_dir):
        create_manual_correction_gui(this_config, DEBUG=DEBUG)
