import os
import zarr
import napari
from DLC_for_WBFM.utils.projects.utils_project import safe_cd


def napari_of_training_data(project_dir):
    # TODO: read from config file, not project directory
    training_dat_fname = os.path.join('2-training', 'training_data_red_channel.zarr')
    training_seg_fname = os.path.join('2-training', 'reindexed_masks.zarr')
    with safe_cd(project_dir):
        z_dat = zarr.open_array(training_dat_fname)
        z_seg = zarr.open_array(training_seg_fname)

    viewer = napari.view_labels(z_seg, ndisplay=3)
    viewer.view_data(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg


def napari_of_full_data(project_dir):
    # TODO: read from config file, not project directory
    training_dat_fname = os.path.join('4-traces', 'data_red_channel.zarr')
    dat_exists = os.path.exists(training_dat_fname)
    training_seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')
    with safe_cd(project_dir):
        if dat_exists:
            z_dat = zarr.open_array(training_dat_fname)
        z_seg = zarr.open_array(training_seg_fname)

    viewer = napari.view_labels(z_seg, ndisplay=3)
    if dat_exists:
        viewer.view_data(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg