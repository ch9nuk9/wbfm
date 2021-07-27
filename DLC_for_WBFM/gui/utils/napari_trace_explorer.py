import argparse
from dataclasses import dataclass
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
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


@dataclass
class finished_project_data:
    project_dir: str

    red_data: zarr.Array
    green_data: zarr.Array

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    dlc_raw: pd.DataFrame

    raw_segmentation: zarr.Array
    segmentation: zarr.Array

    behavior_annotations: pd.DataFrame
    background_per_pixel: float

    @staticmethod
    def unpack_config_file(project_path):
        project_dir = Path(project_path).parent
        cfg = load_config(project_path)
        with safe_cd(project_dir):
            traces_cfg = load_config(cfg['subfolder_configs']['traces'])
            segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
            tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])

        return cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def load_data_from_configs(cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir):
        red_dat_fname = cfg['preprocessed_red']
        green_dat_fname = cfg['preprocessed_green']
        red_traces_fname = traces_cfg['traces']['red']
        green_traces_fname = traces_cfg['traces']['green']
        dlc_raw_fname = tracking_cfg['final_3d_tracks']['df_fname']
        seg_fname_raw = segment_cfg['output']['masks']
        seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')

        fname = r"3-tracking\postprocessing\manual_behavior_annotation.xlsx"  # TODO

        red_data = zarr.open(red_dat_fname)
        green_data = zarr.open(green_dat_fname)

        with safe_cd(project_dir):
            red_traces = pd.read_hdf(red_traces_fname)
            green_traces = pd.read_hdf(green_traces_fname)
            dlc_raw = pd.read_hdf(dlc_raw_fname)

            # Segmentation
            if '.zarr' in seg_fname_raw:
                raw_segmentation = zarr.open(seg_fname_raw, mode='r')
            else:
                raw_segmentation = None

            if os.path.exists(seg_fname):
                segmentation = zarr.open(seg_fname, mode='r')
            else:
                segmentation = None

            behavior_annotations = pd.read_excel(fname, sheet_name='behavior')['Annotation']

        # TODO: do not hardcode
        background_per_pixel = 15

        start = cfg['dataset_params']['start_volume']
        end = start + cfg['dataset_params']['num_frames']
        x = list(range(start, end))

        # Return a full object
        obj = finished_project_data(
            project_dir,
            red_data,
            green_data,
            red_traces,
            green_traces,
            dlc_raw,
            raw_segmentation,
            segmentation,
            behavior_annotations,
            background_per_pixel
        )

        return obj

    @staticmethod
    def load_all_project_data_from_config(project_path):
        args = finished_project_data.unpack_config_file(project_path)
        return finished_project_data.load_data_from_configs(*args)




class napari_trace_explorer(QtWidgets.QWidget):

    def __init__(self, project_path):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)

        ########################
        # Load Configs
        ########################
        self.dat = finished_project_data.load_all_project_data_from_config(project_path)

        # out = finished_project_data.unpack_config_file(project_path)
        # cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir = out
        # self._load_data(cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir)

        # Temporary
        # self.df = self.dlc_raw

    # def _load_data(self, cfg, segment_cfg, tracking_cfg, traces_cfg, project_dir):
    #     ########################
    #     # Load Data
    #     ########################
    #     # Raw data
    #     red_dat_fname = cfg['preprocessed_red']
    #     green_dat_fname = cfg['preprocessed_green']
    #     self.red_data = zarr.open(red_dat_fname)
    #     self.green_data = zarr.open(green_dat_fname)
    #
    #     with safe_cd(self.project_dir):
    #         # Traces
    #         red_traces_fname = traces_cfg['traces']['red']
    #         green_traces_fname = traces_cfg['traces']['green']
    #         dlc_raw_fname = tracking_cfg['final_3d_tracks']['df_fname']
    #         self.red_traces = pd.read_hdf(red_traces_fname)
    #         self.green_traces = pd.read_hdf(green_traces_fname)
    #         self.dlc_raw = pd.read_hdf(dlc_raw_fname)
    #
    #         # Segmentation
    #         seg_fname_raw = segment_cfg['output']['masks']
    #         if '.zarr' in seg_fname_raw:
    #             self.raw_segmentation = zarr.open(seg_fname_raw, mode='r')
    #         else:
    #             self.raw_segmentation = None
    #
    #         seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')
    #         if os.path.exists(seg_fname):
    #             self.segmentation = zarr.open(seg_fname, mode='r')
    #         else:
    #             self.segmentation = None
    #
    #         fname = r"3-tracking\postprocessing\manual_behavior_annotation.xlsx"
    #         self.behavior_annotations = pd.read_excel(fname, sheet_name='behavior')['Annotation']
    #
    #     # TODO: do not hardcode
    #     self.background_per_pixel = 15
    #
    #     start = cfg['dataset_params']['start_volume']
    #     end = start + cfg['dataset_params']['num_frames']
    #     self.x = list(range(start, end))


    # def _load_config_files(self, project_config):
    #     self.project_dir = Path(project_config).parent
    #     cfg = load_config(project_config)
    #     self.cfg = cfg
    #     with safe_cd(self.project_dir):
    #         traces_cfg = load_config(cfg['subfolder_configs']['traces'])
    #         self.traces_cfg = traces_cfg
    #         segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
    #         self.segment_cfg = segment_cfg
    #         tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])
    #         self.tracking_cfg = tracking_cfg
    #
    #     return cfg, traces_cfg, tracking_cfg

    def setupUi(self, viewer: napari.Viewer):

        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = list(self.dat.dlc_raw.columns.levels[0])
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
        self.update_trace_subplot()

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
            change_viewer_time_point(viewer, dt=1, a_max=len(self.dat.dlc_raw) - 1)
            zoom_using_viewer(viewer, layer_name='deeplabcut_track', zoom=None)

        @viewer.bind_key(',', overwrite=True)
        def zoom_previous(viewer):
            change_viewer_time_point(viewer, dt=-1, a_max=len(self.dat.dlc_raw) - 1)
            zoom_using_viewer(viewer, layer_name='deeplabcut_track', zoom=None)

    def initialize_trace_subplot(self):
        self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.static_ax = self.mpl_widget.figure.subplots()
        self.trace_line = self.static_ax.plot(self.calculate_trace())[0]
        # self.time_line = self.static_ax.vlines(*self.calculate_time_line())
        self.time_line = self.static_ax.plot(*self.calculate_time_line())[0]
        self.color_using_behavior()
        self.connect_time_line_callback()

        self.viewer.window.add_dock_widget(self.mpl_widget, area='bottom')

    def update_trace_subplot(self):
        self.trace_line.set_ydata(self.calculate_trace())
        # t, y0, y1, _ = self.calculate_time_line()
        # self.time_line.set_segments(np.array([[t, y0], [t, y1]]))
        self.time_line.set_data(self.calculate_time_line()[:2])
        self.color_using_behavior()
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
            line_color = 'b'
        else:
            # title = "Tracking lost!"
            line_color = 'r'
        # print(f"Calculated vertical line for t={t}")
        return [t, t], [ymin, ymax], line_color
        # return t, ymin, ymax, line_color
        # self.time_line.update_line(t, ymin, ymax, line_color)

    def calculate_trace(self):
        # i = self.changeNeuronsDropdown.currentIndex()
        i = self.current_name

        g = self.dat.green_traces
        r = self.dat.red_traces
        # print(df)
        g_raw = g[i]['brightness']
        r_raw = r[i]['brightness']
        bg = self.dat.background_per_pixel * g[i]['volume']

        smoothing_func = lambda x: x
        # y = smoothing_func((g_raw - bg)/(r_raw - bg))
        y = smoothing_func(g_raw - bg)
        self.y = y
        return y

    def get_track_data(self):
        self.current_name = self.changeNeuronsDropdown.currentText()
        return self.build_tracks_from_name()

    def color_using_behavior(self):
        shade_using_behavior(self.dat.behavior_annotations)

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

        self.dat.dlc_raw = self.dat.dlc_raw.drop(columns=self.current_name, level=0)
        self.dat.dlc_raw = pd.concat([self.dat.dlc_raw, new_df], axis=1)

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_name
        new_points = self.viewer.layers['deeplabcut_track'].data

        col = pd.MultiIndex.from_product([[self.current_name], ['z', 'x', 'y', 'likelihood']])
        df_new = pd.DataFrame(columns=col, index=self.dat.dlc_raw.index)

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
        zxy_array = np.array(self.dat.dlc_raw[self.current_name][coords])
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)
        # Remove low likelihood
        to_remove = self.dat.dlc_raw[self.current_name]['likelihood'] < likelihood_thresh
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
    viewer.add_image(ui.dat.red_data, name="Red data", opacity=0.5)
    viewer.add_image(ui.dat.green_data, name="Green data", opacity=0.5)
    viewer.add_labels(ui.dat.raw_segmentation, name="Raw segmentation", opacity=0.5)
    if ui.dat.segmentation is not None:
        viewer.add_labels(ui.dat.segmentation, name="Colored segmentation")

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