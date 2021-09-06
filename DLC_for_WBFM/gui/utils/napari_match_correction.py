from pathlib import Path

import napari
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import win32com.client

from DLC_for_WBFM.gui.utils.utils_gui import zoom_using_viewer, change_viewer_time_point
from DLC_for_WBFM.utils.projects.finished_project_data import finished_project_data
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_labels_from_traces_dataframe, \
    napari_labels_from_frames
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior


class napari_track_correction(QtWidgets.QWidget):

    def __init__(self, project_path, excel_path):
        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dat = finished_project_data.load_final_project_data_from_config(project_path)
        self.excel_path = excel_path

        # Create an instance of the Excel Application & make it visible.
        self.excel_app = win32com.client.Dispatch('Excel.Application')
        self.excel_app.Visible = True

        self.excel_workbook = self.excel_app.Workbooks.Open(excel_path)


    # def setupUi(self, viewer: napari.Viewer):
    #
    #     # Load dataframe and path to outputs
    #     self.viewer = viewer
    #
    #     # Load excel matching data and start workbook

    def initialize_shortcuts(self):
        # viewer = self.viewer
        pass
        # @viewer.bind_key('.', overwrite=True)
        # def zoom_next(viewer):
        #     change_viewer_time_point(viewer, dt=1, a_max=len(self.dat.final_tracks) - 1)
        #     zoom_using_viewer(viewer, layer_name='final_track', zoom=None)
        #
        # @viewer.bind_key(',', overwrite=True)
        # def zoom_previous(viewer):
        #     change_viewer_time_point(viewer, dt=-1, a_max=len(self.dat.final_tracks) - 1)
        #     zoom_using_viewer(viewer, layer_name='final_track', zoom=None)

    def closeEvent(self, event):
        # Close Excel
        self.excel_workbook.close(True)
        self.excel_app.Quit()

        # and afterwards call the closeEvent of the super-class
        super(self, self).closeEvent(event)


def build_napari_match_corrector(project_config, excel_path, DEBUG=False):
    viewer = napari.Viewer(ndisplay=2)

    # Build object that has all the data
    ui = napari_track_correction(project_config, excel_path)

    # Build Napari and add widgets
    print("Finished loading data, starting napari...")
    viewer.add_image(ui.dat.red_data, name="Red data", opacity=0.5, colormap='red', visible=False)
    viewer.add_labels(ui.dat.raw_segmentation, name="Raw segmentation", opacity=0.4, visible=False)
    if ui.dat.segmentation is not None:
        viewer.add_labels(ui.dat.segmentation, name="Colored segmentation", opacity=0.4)

    # Add a text overlay: new IDs
    df = ui.dat.red_traces
    options = napari_labels_from_traces_dataframe(df)
    viewer.add_points(**options)

    # Add a text overlay: original IDs
    frames = ui.dat.raw_frames
    options = napari_labels_from_frames(frames, num_frames=1)
    viewer.add_points(**options)

    # Actually dock my additional gui elements
    # ui.setupUi(viewer)
    # viewer.window.add_dock_widget(ui)
    # ui.show()

    print("Finished GUI setup")

    napari.run()
