import concurrent
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets

from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
import napari
from DLC_for_WBFM.gui.utils.napari_trace_explorer import napari_trace_explorer
from DLC_for_WBFM.utils.postprocessing.utils_imputation import get_closest_tracklet_to_point


def _init_tracklets(project_data):
    return project_data.df_all_tracklets


def main():
    fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-imputation/project_config.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname, to_load_tracklets=True)

    app = QtWidgets.QApplication(sys.argv)
    viewer = napari.Viewer(ndisplay=3)
    ui, viewer = napari_trace_explorer(project_data, viewer=viewer)

    use_tracklets = False
    if use_tracklets:
        df_tracklets = project_data.df_all_tracklets

        # project_data.add_layers_to_viewer(viewer)
        # print(df_tracklets)
        # print(df_tracklets.iloc[0,:])
        # all_zxy = np.reshape(df_tracklets.iloc[0, :].to_numpy(), (-1, 4))
        # all_zxy = all_zxy[~np.isnan(all_zxy).any(axis=1)][:, :3]
        # print(all_zxy)

        seg_layer = viewer.layers['Raw segmentation']

        max_dist = 10.0

        @seg_layer.mouse_drag_callbacks.append
        def on_click(layer, event):
            # start_point, _ = layer.get_ray_intersections(
            #     position=event.position,
            #     view_direction=event.view_direction,
            #     dims_displayed=event.dims_displayed,
            #     world=True
            # )
            seg_index = layer.get_value(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True
            )
            print(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                  f"and position {event.position[1:]}")

            dist, ind, name = get_closest_tracklet_to_point(
                i_time=int(event.position[0]),
                target_pt=event.position[1:],
                df_tracklets=df_tracklets,
                verbose=2
            )
            dist = dist[0][0]
            print(f"Neuron is part of tracklet {name} with distance {dist}")

            if dist < max_dist:
                df_single_track = df_tracklets[name]
                print(f"Adding tracklet of length {df_single_track['z'].count()}")
                all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track)
                viewer.add_tracks(track_of_point, name=name)

                print(df_single_track.dropna(inplace=False))
            else:
                print(f"Tracklet too far away; not adding")

    napari.run()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
