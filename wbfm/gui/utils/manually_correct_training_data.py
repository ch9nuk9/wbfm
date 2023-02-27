import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr
from PyQt5 import QtWidgets

from wbfm.gui.utils.utils_gui import zoom_using_layer_in_viewer, change_viewer_time_point
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.tracklets.training_data_from_tracklets import get_or_recalculate_which_frames


class manual_annotation_widget(QtWidgets.QWidget):

    def __init__(self):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)

    def setupUi(self, df: pd.DataFrame, output_dir: str, viewer: napari.Viewer, annotation_output_name: str):
        # Load dataframe and path to outputs
        self.annotation_output_name = annotation_output_name
        self.viewer = viewer
        self.output_dir = output_dir
        self.df = df
        neuron_names = get_names_from_df(df)
        self.current_name = neuron_names[0]

        # Change neurons (dropdown)
        self.changeNeuronsButton = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.changeNeuronsButton.addItems(neuron_names)
        self.changeNeuronsButton.setItemText(0, self.current_name)
        self.changeNeuronsButton.currentIndexChanged.connect(self.change_neurons)
        self.verticalLayout.addWidget(self.changeNeuronsButton)

        # Save annotations (button)
        self.saveButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.saveButton.clicked.connect(self.save_annotations)
        self.saveButton.setText("Save Annotations")
        self.verticalLayout.addWidget(self.saveButton)

        self.initialize_track_layers()
        self.initialize_shortcuts()

    def change_neurons(self):
        self.update_dataframe_using_points()
        self.update_track_layers()

    def update_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()
        self.viewer.layers['pts_with_future_and_past'].data = point_layer_data
        self.viewer.layers['track_of_point'].data = track_layer_data

        zoom_using_layer_in_viewer(self.viewer)

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        self.viewer.add_points(point_layer_data, name="pts_with_future_and_past", n_dimensional=True, symbol='cross',
                               **points_opt)
        self.viewer.add_tracks(track_layer_data, name="track_of_point")

        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('.', overwrite=True)
        def zoom_next(viewer):
            change_viewer_time_point(viewer, dt=1, a_max=len(self.df) - 1)
            zoom_using_layer_in_viewer(viewer, zoom=None)

        @viewer.bind_key(',', overwrite=True)
        def zoom_previous(viewer):
            change_viewer_time_point(viewer, dt=-1, a_max=len(self.df) - 1)
            zoom_using_layer_in_viewer(viewer, zoom=None)

        # @viewer.bind_key('.', overwrite=True)
        # def zoom_next(dummy):
        #     self.zoom_and_change_time(1)
        #
        # @viewer.bind_key(',', overwrite=True)
        # def zoom_previous(dummy):
        #     self.zoom_and_change_time(-1)

    #
    # def zoom_and_change_time(self, dt=0):
    #     viewer = self.viewer
    #     zoom = self.get_zoom()
    #     change_viewer_time_point(viewer, dt)
    #     zoom_using_viewer(viewer, zoom)
    #
    #
    # def get_zoom(self):
    #     return self.changeZoomSlider.value()

    def get_track_data(self):
        self.current_name = self.changeNeuronsButton.currentText()
        return self.build_tracks_from_name()

    def save_annotations(self):
        self.update_dataframe_using_points()
        # self.df[self.current_name] = new_df[self.current_name]

        out_fname = self.annotation_output_name
        self.df.to_hdf(out_fname, 'df_with_missing')

        out_fname = str(Path(out_fname).with_suffix('.csv'))
        #     df_old = pd.read_csv(out_fname)
        #     df_old[name] = df_new[name]
        #     df_old.to_csv(out_fname, mode='a')
        self.df.to_csv(out_fname)  # Just overwrite

        print(f"Saved manual annotations for neuron {self.current_name} at {out_fname}")

    def update_dataframe_using_points(self):
        # print("Before saving:")
        # print(self.df)

        new_df = self.build_df_of_current_points()

        # print("pandas try 1:")
        # self.df[self.current_name] = new_df[self.current_name]
        # print(self.df)

        # print("pandas try 2:")
        self.df = self.df.drop(columns=self.current_name, level=0)
        self.df = pd.concat([self.df, new_df], axis=1)
        # self.df = self.df.join(new_df)
        # print(self.df)

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_name
        new_points = self.viewer.layers['pts_with_future_and_past'].data

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

        # col = pd.MultiIndex.from_product([[self.current_name], ['t', 'z', 'x', 'y', 'likelihood']])
        # col = pd.MultiIndex.from_product([[self.current_name], ['z', 'x', 'y', 'likelihood']])
        # df_new = pd.DataFrame(columns=col, index=self.df.index)
        #
        # # df_new[(name, 't')] = new_points[:, 0]
        # df_new[(name, 'z')] = new_points[:, 1]
        # df_new[(name, 'y')] = new_points[:, 2]
        # df_new[(name, 'x')] = new_points[:, 3]
        # df_new[(name, 'likelihood')] = np.ones(new_points.shape[0])
        # df_new[(name, 'likelihood')] = self.df[(name, 'likelihood')]  # Same as before

        # df_new.sort_values((name, 't'), inplace=True, ignore_index=True)

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


def create_manual_correction_gui(cfg: ModularProjectConfig,
                                 segment_cfg: SubfolderConfigFile,
                                 training_cfg: SubfolderConfigFile,
                                 tracking_cfg: SubfolderConfigFile,
                                 corrector_name='Charlie', initial_annotation_name=None, DEBUG=False):
    """
    Creates a napari-based gui for correcting tracks

    For now, only works with training data
    """
    project_dir = cfg.project_dir

    annotation_output_name = os.path.join(project_dir, '2-training_data', 'manual_tracking',
                                          f'corrected_tracks-{corrector_name}.h5')
    if Path(annotation_output_name).exists():
        raise FileExistsError(f"File already {annotation_output_name} exists! Please rename or delete")

    with safe_cd(project_dir):

        fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
        df_raw_matches = pd.read_pickle(fname)

        # Get the frames chosen as training data, or recalculate
        which_frames = list(
            get_or_recalculate_which_frames(DEBUG, df_raw_matches, cfg.config['dataset_params']['num_frames'],
                                            training_cfg))

        # Import segmentation
        fname = segment_cfg.config['output_masks']
        raw_segmentation = zarr.open(fname)

        fname = os.path.join('2-training_data', 'reindexed_masks.zarr')
        if Path(fname).exists():
            colored_segmentation = zarr.open(fname)
        else:
            colored_segmentation = None

        if initial_annotation_name is None:
            # Use the output of my tracker
            fname = training_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
            df_initial_annotations = pd.read_hdf(fname)
            # fname = os.path.join('2-training_data', 'training_data_tracks.h5')
            # Future: not hardcoded experimenter
            # df = pd.read_hdf(fname)['Charlie'].copy()
        else:
            # Use partially manually annotated tracking
            df_initial_annotations = pd.read_hdf(initial_annotation_name)

        # Import raw data
        fname = cfg.resolve_relative_path_from_config('preprocessed_red')
        red_data = zarr.open(fname)

    print("Finished loading data, starting napari...")

    # Build Napari and add widgets
    viewer = napari.view_image(red_data[which_frames[0]:which_frames[-1] + 1, ...], name="Red data", ndisplay=2,
                               opacity=0.5)
    viewer.add_labels(raw_segmentation[which_frames[0]:which_frames[-1] + 1, ...], name="Raw segmentation", opacity=0.5)
    if colored_segmentation is not None:
        viewer.add_labels(colored_segmentation)

    output_dir = os.path.join("2-training_data", "manual_tracking")
    ui = manual_annotation_widget()
    ui.setupUi(df_initial_annotations, output_dir, viewer, annotation_output_name)

    # Actually dock
    viewer.window.add_dock_widget(ui)
    ui.show()

    print("Finished GUI setup")

    napari.run()
