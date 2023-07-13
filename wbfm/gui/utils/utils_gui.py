import logging
import os
import subprocess
import sys

import napari
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QComboBox, QVBoxLayout, QWidget, QApplication, QMessageBox

from wbfm.utils.general.postprocessing.base_cropping_utils import get_crop_coords3d
from wbfm.utils.general.video_and_data_conversion.import_video_as_array import get_single_volume, \
    get_single_volume_specific_slices


def get_cropped_frame(fname, t, num_slices, zxy, crop_sz, to_flip=False):
    # print(f"Loading file: {fname}")
    # print(f"Location: {zxy}")
    if crop_sz is not None:
        crop_coords = get_crop_coords3d(zxy, crop_sz)
        z, x, y = crop_coords
        if len(z) > 1:
            start_slice, end_slice = z[0], z[-1]
        else:
            start_slice, end_slice = z[0], z[0] + 1
        dat = get_single_volume_specific_slices(fname, t, num_slices,
                                                start_slice, end_slice)
        if to_flip:
            dat = np.flip(dat, axis=1)
        # print(f"crop_coords: {crop_coords}")
        dat_crop = dat[x[0]:x[-1], y[0]:y[-1]]
    else:
        dat_crop = get_single_volume(fname, t, num_slices, dtype='uint16')
        if to_flip:
            dat_crop = np.flip(dat_crop, axis=2)

    return _fix_dimension_for_plt(crop_sz, dat_crop)


def _fix_dimension_for_plt(crop_sz, dat_crop):
    # Final output should be XYC
    if len(dat_crop.shape) == 3:
        if crop_sz is None:
            # Just visualize center of worm
            dat_crop = dat_crop[15]  # Remove z
        else:
            dat_crop = dat_crop[0]
    return np.array(dat_crop)


def get_crop_from_zarr(zarr_array, t, zxy, crop_sz, force_2d=False):
    if crop_sz is not None:
        crop_coords = get_crop_coords3d(zxy, crop_sz)
        z, x, y = crop_coords
        if len(z) > 1:
            start_slice, end_slice = z[0], z[-1] + 1
        else:
            start_slice, end_slice = z[0], z[0] + 1
        # print(f"Zarr size before crop: {zarr_array.shape}")
        this_volume = np.array(zarr_array[t, ...])
        dat_crop = this_volume[start_slice:end_slice, x[0]:x[-1]+1, y[0]:y[-1]+1]
        # print(f"Zarr size after crop: {dat_crop.shape}")
    else:
        dat_crop = zarr_array[t, :, :, :]

    if force_2d:
        return _fix_dimension_for_plt(crop_sz, dat_crop)
    else:
        return dat_crop


def array2qt(img):
    # From: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    h, w, channel = img.shape
    # bytesPerLine = 3 * w
    # return QtGui.QPixmap(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    new_img = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(new_img)


def zoom_using_layer_in_viewer(viewer: napari.Viewer, layer_name='pts_with_future_and_past', zoom=None,
                               layer_is_full_size_and_single_neuron=True, ind_within_layer=None) -> None:
    # Get current point
    t = viewer.dims.current_step[0]
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]

        if layer_is_full_size_and_single_neuron:
            tzxy = get_zxy_from_single_neuron_layer(layer, t)
        else:
            tzxy = get_zxy_from_multi_neuron_layer(layer, t, ind_within_layer)
        print(f"Centering screen using: tzxy={tzxy} from layer {layer}")
    else:
        print(f"Layer {layer_name} not found; no zooming")
        return

    # Enhancement: better way to check for nesting
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    # Data may be actually a null value (but t should be good)
    tzxy[0] = t
    zoom_using_viewer(tzxy, viewer, zoom)


def zoom_using_viewer(tzxy, viewer, zoom):
    try:
        is_positive = tzxy[2] > 0 and tzxy[3] > 0
        is_finite = not all(np.isnan(tzxy))
        # Center to the neuron in xy
        if zoom is not None:
            viewer.camera.zoom = zoom
        if is_positive and is_finite:
            viewer.camera.center = tzxy[1:]
        # Center around the neuron in z
        if is_positive and is_finite:
            viewer.dims.current_step = (tzxy[0], tzxy[1], 0, 0)
    except IndexError:
        logging.warning("Index error in zooming; skipping")
        return


def get_zxy_from_single_neuron_layer(layer, t, ind_within_layer=None):
    return layer.data[t]


def get_zxy_from_multi_neuron_layer(layer, t, ind_within_layer=None):
    # e.g. text labels, with all neurons in a time point in a row (thus t is no longer a direct index)
    # Or, if nans have been dropped from an otherwise full-size layer
    # Note: if ind_within_layer is None, it has no effect
    dat = layer.data
    if dat.shape[1] == 5:
        # Tracks layer; neuron index is now first column
        dat = dat[:, 1:]
    elif dat.shape[1] == 4:
        # Points layer
        pass
    else:
        raise ValueError(f"Unrecognized layer shape {dat.shape}")
    ind = dat[:, 0] == t

    if len(np.where(ind)[0]) == 0:
        fake_dat = np.zeros_like(layer.data[0, :])
        fake_dat[0] = t
        # logging.warning(f"Time {t} not found in layer: {layer.data[:, 0]}")
        return fake_dat
    else:
        if ind_within_layer is not None:
            return layer.data[ind, :][ind_within_layer, :]
        else:
            return layer.data[ind, :]


def change_viewer_time_point(viewer: napari.Viewer,
                             dt: int = None, t_target: int = None, a_max: int = None) -> tuple:
    # Increment time
    if dt is not None:
        t = np.clip(viewer.dims.current_step[0] + dt, a_min=0, a_max=a_max)
    elif t_target is not None:
        t = np.clip(t_target, a_min=0, a_max=a_max)
    else:
        raise ValueError("Must pass either target time or dt")
    tzxy = (t,) + viewer.dims.current_step[1:]
    viewer.dims.current_step = tzxy

    return tzxy


def add_fps_printer(viewer):
    # From: https://github.com/napari/napari/issues/836
    def fps_status(viewer, x):
        # viewer.help = f'{x:.1f} frames per second'
        print(f'{x:.1f} frames per second')

    viewer.window.qt_viewer.canvas.measure_fps(callback=lambda x: fps_status(viewer, x))


def build_gui_for_grid_plots(parent_folder, DEBUG=False):
    # Build a GUI for selecting which grid plots to view
    # Each grid plot is a png file
    # Subfolder structure is:
    # parent_folder
    #  - folder of projects
    #    - folder of single project
    #      - folder called "4-traces"
    #        - individual png files
    # There can be many png files, and we want a dropdown to select which one to view

    # Get all the project parent folders
    project_parent_folders = [os.path.join(parent_folder, x) for x in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, x))]
    # Get individual project_folders as a nested dictionary of {project_parent_folder: {project_name: project_folder}}
    project_folders = {}
    for project_parent_folder in project_parent_folders:
        # Set the outer key as just the folder name
        key = os.path.basename(project_parent_folder)
        project_folders[key] = {os.path.basename(x): os.path.join(project_parent_folder, x)
                                    for x in os.listdir(project_parent_folder)
                                    if os.path.isdir(os.path.join(project_parent_folder, x))}

    # Remove any folders that have not subfolders
    project_folders = {k: v for k, v in project_folders.items() if len(v) > 0}

    # Get all the png files as a nested dictionary of {project_parent_folder: {project_name: {png_basename: png_full_path}}}
    png_files = {}
    for project_parent_folder, project_name_dict in project_folders.items():
        png_files[project_parent_folder] = {}
        for project_name, project_full_path in project_name_dict.items():
            this_subfolder = os.path.join(project_full_path, "4-traces")
            # There could be other folders with different subfolders, so skip if this isn't a folder
            if not os.path.isdir(this_subfolder):
                continue
            png_basename = [x for x in os.listdir(this_subfolder) if x.endswith(".png")]
            png_full_path = [os.path.join(this_subfolder, x) for x in png_basename]
            # Do not save if there are no png files
            if len(png_basename) > 0:
                png_files[project_parent_folder][project_name] = dict(zip(png_basename, png_full_path))

    # Remove any folders that have no subfolders
    png_files = {k: v for k, v in png_files.items() if len(v) > 0}
    # Remove subfolders that have no png files
    for k, v in png_files.items():
        png_files[k] = {k2: v2 for k2, v2 in v.items() if len(v2) > 0}

    # Build the GUI using qtwidgets
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    window.setLayout(layout)
    # Add a dropdown to select the folder of projects
    project_parent_dropdown = QComboBox()
    project_parent_dropdown.addItems(png_files.keys())
    layout.addWidget(project_parent_dropdown)

    # Add a dropdown to select the project within the folder
    project_dropdown = QComboBox()
    layout.addWidget(project_dropdown)

    # Add a dropdown to select the png file
    png_dropdown = QComboBox()
    layout.addWidget(png_dropdown)
    # Add a button to view the selected png file
    view_button = QPushButton("View")
    layout.addWidget(view_button)
    # Add a popup if the image isn't found
    popup = QMessageBox()
    popup.setWindowTitle("Error")
    popup.setText("Image not found")
    popup.setIcon(QMessageBox.Critical)
    popup.setStandardButtons(QMessageBox.Ok)

    # Callback to update the png dropdown when the project dropdown is changed
    def update_png_dropdown():
        project_name = project_dropdown.currentText()
        project_parent_name = project_parent_dropdown.currentText()
        # Do not try if no project is selected
        if project_name == "" or project_parent_name == "":
            return
        # print("Updating png dropdown for: ", project_parent_name, project_name)
        keys = list(png_files[project_parent_name][project_name].keys())
        png_dropdown.clear()
        png_dropdown.addItems(keys)
        # Set the default value as an item that contains the word 'beh'
        for i, key in enumerate(keys):
            if "beh" in key.lower():
                png_dropdown.setCurrentIndex(i)
                break

    # Callback to view the selected png file
    def view_png():
        project_parent_name = project_parent_dropdown.currentText()
        project_name = project_dropdown.currentText()
        png_name = png_dropdown.currentText()
        # Do not try if no file is selected
        if png_name == "" or project_name == "" or project_parent_name == "":
            return
        png_path = png_files[project_parent_name][project_name][png_name]
        # Display the image if it exists using the system default image viewer
        if os.path.exists(png_path):
            print("Opening: ", png_path)
            path = os.path.realpath(png_path)
            if sys.platform == "win32":
                os.startfile(path)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, path])
        else:
            popup.exec()

    # Callback to update the project dropdown when the project parent dropdown is changed
    def update_project_dropdown():
        project_parent_folder = project_parent_dropdown.currentText()
        keys = list(png_files[project_parent_folder].keys())
        project_dropdown.clear()  # Note that this triggers the update_png_dropdown callback
        project_dropdown.addItems(keys)
        project_dropdown.setCurrentIndex(0)
        # print("Updating project dropdown for: ", project_parent_folder)
        # print(png_files[project_parent_folder])

    # Connect the callbacks, ensuring that the parent folder is updated first
    project_parent_dropdown.currentTextChanged.connect(update_project_dropdown)
    project_dropdown.currentTextChanged.connect(update_png_dropdown)
    view_button.clicked.connect(view_png)

    # Set the parent dropdown folder, in order to trigger the callbacks
    project_parent_dropdown.setCurrentIndex(1)
    project_parent_dropdown.setCurrentIndex(0)

    window.show()
    app.exec()


def on_close(self, event, widget):
    # Copied from deeplabcut-napari
    # https://github.com/DeepLabCut/napari-deeplabcut/blob/c05d4a8eb58716da96b97d362519d4ee14e21cac/src/napari_deeplabcut/_widgets.py#L121
    choice = QMessageBox.warning(
        widget,
        "Warning",
        "Data may not be saved. Are you certain you want to quit? "
        "Note: you additionally need to press ctrl-c in the terminal to fully quit the program",
        QMessageBox.Yes | QMessageBox.No,
    )
    if choice == QMessageBox.Yes:
        event.accept()
    else:
        event.ignore()
