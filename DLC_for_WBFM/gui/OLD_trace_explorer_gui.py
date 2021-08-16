# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_raw.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
import os

import zarr
from PyQt5 import QtCore, QtWidgets
import argparse
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from DLC_for_WBFM.gui.utils.utils_gui import get_cropped_frame, get_crop_from_zarr
import pandas as pd
import sys
from pathlib import Path
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import concurrent.futures
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    # See: https://www.learnpyqt.com/tutorials/plotting-matplotlib/
    def __init__(self, parent=None, width=15, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow, project_config, DEBUG):
        # super(MainWindow, self).__init__()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1084, 754)

        ########################
        # Load Configs
        ########################
        cfg, traces_cfg, tracking_cfg = self._load_config_files(project_config)
        ########################
        # Load Data
        ########################
        # For matplotlib
        red_traces_fname = traces_cfg['traces']['red']
        green_traces_fname = traces_cfg['traces']['green']
        dlc_raw_fname = tracking_cfg['final_3d_tracks_df']
        if not DEBUG:
            self.red_traces = pd.read_hdf(red_traces_fname)
            self.green_traces = pd.read_hdf(green_traces_fname)
            self.dlc_raw = pd.read_hdf(dlc_raw_fname)

            seg_fname = self.segment_cfg['output']['masks']
            if '.zarr' in seg_fname:
                seg_fname = os.path.join(self.project_dir, seg_fname)
                # print(f"Opening zarr segmentation at: {seg_fname}")
                self.zarr_array = zarr.open(seg_fname, mode='r')
                # print(f"Size: {self.zarr_array.shape}")
            else:
                self.zarr_array = None

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.dialAndPlotLayout = QtWidgets.QHBoxLayout()
        self.dialAndPlotLayout.setObjectName("dialAndPlotLayout")

        self.dialsLayout = QtWidgets.QVBoxLayout()
        self.dialsLayout.setObjectName("dialsLayout")

        self.plotsLayout = QtWidgets.QVBoxLayout()
        self.plotsLayout.setObjectName("plotsLayout")
        ########################
        # Various selectors
        ########################
        # Neuron selector
        self.neuronSelector = QtWidgets.QComboBox(self.centralwidget)
        self.neuronSelector.setObjectName("neuronSelector")
        neuron_names = traces_cfg['traces']['neuron_names']
        [self.neuronSelector.addItem(name) for name in neuron_names]
        self.dialsLayout.addWidget(self.neuronSelector)
        self.neuronSelector.setItemText(0, neuron_names[0])
        self.neuronSelector.currentIndexChanged.connect(self.update_all_panels)
        # Frame (time) selector
        self.timeSelector = QtWidgets.QSpinBox(self.centralwidget)
        start = cfg['dataset_params']['start_volume']
        self.timeSelector.setMinimum(start)
        end = start + cfg['dataset_params']['num_frames']
        self.timeSelector.setMaximum(end)
        self.timeSelector.setObjectName("timeSelector")
        self.dialsLayout.addWidget(self.timeSelector)
        self.timeSelector.valueChanged.connect(self.update_all_panels)
        # Background selector
        self.backgroundSelector = QtWidgets.QSpinBox(self.centralwidget)
        self.backgroundSelector.setMinimum(0)
        self.backgroundSelector.setValue(15)
        self.backgroundSelector.setMaximum(100)
        self.backgroundSelector.setObjectName("backgroundSelector")
        self.dialsLayout.addWidget(self.backgroundSelector)
        self.backgroundSelector.valueChanged.connect(self.update_only_traces)
        # DLC confidence
        self.confidenceSelector = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.confidenceSelector.setMinimum(0)
        self.confidenceSelector.setValue(0.9)
        self.confidenceSelector.setSingleStep(0.1)
        self.confidenceSelector.setMaximum(1)
        self.confidenceSelector.setObjectName("confidenceSelector")
        self.dialsLayout.addWidget(self.confidenceSelector)
        self.confidenceSelector.valueChanged.connect(self.update_only_traces)
        # Trace (mode) selector (top and bottom)
        self.modeSelectorTop = QtWidgets.QComboBox(self.centralwidget)
        self.modeSelectorTop.setObjectName("modeSelector")
        possible_modes = ['green', 'red', 'ratio']
        [self.modeSelectorTop.addItem(m) for m in possible_modes]
        self.dialsLayout.addWidget(self.modeSelectorTop)
        self.modeSelectorTop.currentIndexChanged.connect(self.update_only_traces)

        self.modeSelectorBottom = QtWidgets.QComboBox(self.centralwidget)
        self.modeSelectorBottom.setObjectName("modeSelector")
        possible_modes = ['green', 'red', 'ratio']
        [self.modeSelectorBottom.addItem(m) for m in possible_modes]
        self.modeSelectorBottom.setCurrentText('ratio')
        self.dialsLayout.addWidget(self.modeSelectorBottom)
        self.modeSelectorBottom.currentIndexChanged.connect(self.update_only_traces)

        # Crop selectors
        self.cropXSelector = QtWidgets.QSpinBox(self.centralwidget)
        self.cropXSelector.setMinimum(10)
        self.cropXSelector.setValue(64)
        self.cropXSelector.setSingleStep(32)
        self.cropXSelector.setMaximum(300)
        self.cropXSelector.setObjectName("cropXSelector")
        self.dialsLayout.addWidget(self.cropXSelector)
        self.cropXSelector.valueChanged.connect(self.update_all_panels)

        self.cropYSelector = QtWidgets.QSpinBox(self.centralwidget)
        self.cropYSelector.setMinimum(10)
        self.cropYSelector.setValue(64)
        self.cropYSelector.setSingleStep(32)
        self.cropYSelector.setMaximum(300)
        self.cropYSelector.setObjectName("cropYSelector")
        self.dialsLayout.addWidget(self.cropYSelector)
        self.cropYSelector.valueChanged.connect(self.update_all_panels)

        self.dialAndPlotLayout.addLayout(self.dialsLayout)
        ########################
        # Traces (matplotlib)
        ########################
        self.tracesPlotTop = MplCanvas(self, width=15, height=4, dpi=100)
        self.tracesPlotBottom = MplCanvas(self, width=15, height=4, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        # ENHANCE: add a toolbar
        # toolbar = NavigationToolbar(sc, self)
        # self.verticalLayout_plot = QtWidgets.QVBoxLayout()
        # self.verticalLayout_plot.addWidget(toolbar)
        # self.verticalLayout_plot.addWidget(sc)
        # self.horizontalLayout_4.addLayout(self.verticalLayout_plot)

        self.plotsLayout.addWidget(self.tracesPlotTop)
        self.plotsLayout.addWidget(self.tracesPlotBottom)

        self.dialAndPlotLayout.addLayout(self.plotsLayout)
        self.verticalLayout_4.addLayout(self.dialAndPlotLayout)

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")

        ########################
        # Segmentation
        ########################
        # sc = MplCanvas(self, width=5, height=4, dpi=100)
        # self.segFocusPlt = sc
        # self.horizontalLayout_6.addWidget(self.segFocusPlt)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.segPlt = sc
        self.horizontalLayout_6.addWidget(self.segPlt)

        ########################
        # Red channel
        ########################
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.redPlt = sc
        self.horizontalLayout_6.addWidget(self.redPlt)

        ########################
        # Green channel
        ########################
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.greenPlt = sc
        self.horizontalLayout_6.addWidget(self.greenPlt)

        self.horizontalLayout_5.addLayout(self.horizontalLayout_6)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 1084, 22))
        # self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.update_all_panels()

    def _load_config_files(self, project_config):
        cfg = load_config(project_config)
        self.project_dir = Path(project_config).parent
        self.cfg = cfg
        traces_cfg = load_config(cfg['subfolder_configs']['traces'])
        self.traces_cfg = traces_cfg
        segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])
        self.segment_cfg = segment_cfg
        tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])
        self.tracking_cfg = tracking_cfg
        self.crop_sz = (1, 48, 48)
        start = cfg['dataset_params']['start_volume']
        end = start + cfg['dataset_params']['num_frames']
        self.x = list(range(start, end))
        return cfg, traces_cfg, tracking_cfg


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # self.neuronSelector.setItemText(0, _translate("MainWindow", "neuron0"))
        # self.tracesPlt.setText(_translate("MainWindow", "This is the very very large matplotlib \'label\' of the traces over time"))
        # self.segmentationImg.setText(_translate("MainWindow", "TextLabel"))
        # self.redChannelImg.setText(_translate("MainWindow", "TextLabel"))
        # self.greenChannelImg.setText(_translate("MainWindow", "TextLabel"))

    def update_all_panels(self):
        with safe_cd(self.project_dir):
            self.update_current_centroid()
            self.update_crop()

            update_funcs = [self.update_only_traces]
            # self.update_only_traces()
            if not self.tracking_lost:
                update_funcs.extend([self.update_segmentation, self.update_red, self.update_green])
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = executor.map(lambda f: f(), update_funcs)

    def update_only_traces(self):
        with safe_cd(self.project_dir):
            self.update_current_traces()
            self.update_traces_plot()

    def update_current_traces(self):
        mode_top = self.modeSelectorTop.currentText()
        mode_bottom = self.modeSelectorBottom.currentText()
        current_neuron = self.neuronSelector.currentText()
        self.current_traces_top = self._get_trace_from_mode(current_neuron, mode_top)
        self.current_traces_bottom = self._get_trace_from_mode(current_neuron, mode_bottom)
        self.remove_low_conf_values()

    def _get_trace_from_mode(self, current_neuron, mode):
        if mode == 'green':
            y_raw = self.green_traces[current_neuron]['brightness']
            y = y_raw / self.green_traces[current_neuron]['volume']
        elif mode == 'red':
            y_raw = self.red_traces[current_neuron]['brightness']
            y = y_raw / self.red_traces[current_neuron]['volume']
        elif mode == 'ratio':
            red = self.red_traces[current_neuron]['brightness']
            green = self.green_traces[current_neuron]['brightness']
            vol = self.red_traces[current_neuron]['volume']
            back = self.backgroundSelector.value()
            y = (green - back * vol) / (red - back * vol)
        else:
            print(f"Unknown mode ({mode})")
            raise NotImplementedError
        return y

    def remove_low_conf_values(self):
        current_neuron = self.neuronSelector.currentText()
        tracking_confidence = self.dlc_raw[current_neuron]['likelihood']
        threshold = self.confidenceSelector.value()
        ind = (tracking_confidence < threshold)
        self.current_traces_top[ind] = np.nan
        self.current_traces_bottom[ind] = np.nan

    def update_traces_plot(self):
        mode = self.modeSelectorTop.currentText()
        y = self.current_traces_top
        canvas = self.tracesPlotTop.fig.canvas
        title = self._update_canvas(canvas, mode, y)
        self.tracesPlotTop.axes.set_title(title)
        canvas.draw()

        mode = self.modeSelectorBottom.currentText()
        y = self.current_traces_bottom
        canvas = self.tracesPlotBottom.fig.canvas
        title = self._update_canvas(canvas, mode, y)
        self.tracesPlotBottom.axes.set_title(title)
        canvas.draw()

    def _update_canvas(self, canvas, mode, y):
        current_neuron = self.neuronSelector.currentText()
        # Vertical line for time
        t = self.timeSelector.value()
        ymin, ymax = np.min(y), np.max(y)
        # Actually plot
        canvas.axes.cla()  # Clear the canvas.
        canvas.axes.plot(self.x, y, 'k')
        if not self.tracking_lost:
            z, x, y = self.current_centroid
            title = f"{current_neuron}: {mode} trace at ({z:.1f}, {x:.0f}, {y:.0f})"
            line_color = 'b'
        else:
            title = "Tracking lost!"
            line_color = 'r'
        canvas.axes.vlines(t, ymin, ymax, line_color)
        return title

    def update_current_centroid(self):
        current_neuron = self.neuronSelector.currentText()
        t = self.timeSelector.value()
        # TODO: fix x-y switch
        z = self.red_traces[current_neuron]['z_dlc'].loc[t]
        x = self.red_traces[current_neuron]['x_dlc'].loc[t]
        y = self.red_traces[current_neuron]['y_dlc'].loc[t]
        if any([pd.isna(val) for val in [z, x, y]]):
            self.tracking_lost = True
            # Keep last centroid
        else:
            self.current_centroid = (z, x, y)
            self.tracking_lost = False

    def update_crop(self):
        x = self.cropXSelector.value()
        y = self.cropYSelector.value()
        self.crop_sz = (1, x, y)

    def update_segmentation(self):
        t = self.timeSelector.value()
        frame = self.seg_frame_factory(t, self.current_centroid)

        ax = self.segPlt.fig.canvas.axes
        ax.imshow(frame)
        title = "Segmentatation"  # at centroid {self.current_centroid}"
        ax.set_title(title)

        # Also display only the current neuron
        # ax = self.segFocusPlt.fig.canvas.axes
        # ax.imshow(frame)
        # current_neuron = self.neuronSelector.currentText()
        # title = f"Neuron {current_neuron}"  # at centroid {self.current_centroid}"
        # ax.set_title(title)

        self.segPlt.fig.canvas.draw()

    def update_red(self):
        t = self.timeSelector.value()
        frame = self.red_frame_factory(t, self.current_centroid)
        ax = self.redPlt.fig.canvas.axes
        ax.imshow(frame)
        title = "Red Channel"
        ax.set_title(title)
        self.redPlt.fig.canvas.draw()

    def update_green(self):
        t = self.timeSelector.value()
        zxy = self.current_centroid
        frame = self.green_frame_factory(t, zxy)
        ax = self.greenPlt.fig.canvas.axes
        ax.imshow(frame)
        title = "Green Channel"
        ax.set_title(title)
        self.greenPlt.fig.canvas.draw()

    def red_frame_factory(self, t, zxy):
        fname = self.cfg['red_bigtiff_fname']
        num_slices = self.cfg['dataset_params']['num_slices']
        crop_frame = get_cropped_frame(fname, t, num_slices, zxy, self.crop_sz)
        return crop_frame

    def green_frame_factory(self, t, zxy):
        fname = self.cfg['green_bigtiff_fname']
        num_slices = self.cfg['dataset_params']['num_slices']
        sz = self.crop_sz
        to_flip = self.cfg['dataset_params']['red_and_green_mirrored']
        return get_cropped_frame(fname, t, num_slices, zxy, sz, to_flip)

    def seg_frame_factory(self, t, zxy):
        fname = self.segment_cfg['output']['masks']
        num_slices = self.cfg['dataset_params']['num_slices']
        t -= self.cfg['dataset_params']['start_volume']
        if '.zarr' not in fname:
            crop_frame = get_cropped_frame(fname, t, num_slices, zxy, self.crop_sz)
        else:
            crop_frame = get_crop_from_zarr(self.zarr_array, t, zxy, self.crop_sz)
        # TODO: special color for matched neuron
        return crop_frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('project_config',
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='path to config file')
    args = parser.parse_args()

    # Basic setup
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    # Get project settings
    project_config = args.project_config
    # Actually build window
    with safe_cd(Path(project_config).parent):
        ui.setupUi(MainWindow, project_config, args.DEBUG)
        MainWindow.show()
    sys.exit(app.exec_())
