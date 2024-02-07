import cProfile

import napari

from wbfm.gui.utils.utils_gui import change_viewer_time_point
from wbfm.utils.projects.finished_project_data import ProjectData


def basic_time_changes(viewer):
    for t in range(100):
        change_viewer_time_point(viewer, t_target=t)


def profile_napari():
    fname = "/home/charles/dlc_stacks/worm1_for_students/project_config-workstation.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname)

    viewer = project_data.add_layers_to_viewer(which_layers=['Red data'])

    for _ in range(10):
        basic_time_changes(viewer)

    # # Context manager requires python 3.8
    # with cProfile.Profile() as pr:
    #     basic_time_changes(viewer)
    # pr.dump_stats("basic_time_changes.pstat")

    # Second time around
    # with cProfile.Profile() as pr:
    #     basic_time_changes(viewer)
    # pr.dump_stats("basic_time_changes2.pstat")


def profile_napari_ram():
    fname = "/home/charles/dlc_stacks/worm1_for_students/project_config-workstation.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname)

    viewer = napari.Viewer(ndisplay=3)
    dat = project_data.red_data[:100]
    viewer.add_image(dat)

    for _ in range(10):
        basic_time_changes(viewer)


if __name__ == '__main__':
    profile_napari()
    # profile_napari_ram()
