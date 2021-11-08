import numpy as np

from DLC_for_WBFM.gui.utils.utils_gui import build_tracks_from_dataframe
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
import napari
from DLC_for_WBFM.gui.utils.napari_trace_explorer import napari_trace_explorer
from DLC_for_WBFM.utils.postprocessing.utils_imputation import get_closest_tracklet_to_point

def main():
    fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-multiple_templates/project_config.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname)

    viewer = napari.Viewer(ndisplay=3)
    project_data.add_layers_to_viewer(viewer)
    df_tracklets = project_data.df_all_tracklets

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
        # print(event.position[1:], start_point)
        print(f"Event triggered on segmentation: {seg_index}")

        dist, ind, name = get_closest_tracklet_to_point(
            i_time=int(event.position[0]),
            target_pt=event.position[1:],
            df_tracklets=df_tracklets
        )
        dist = dist[0][0]
        print(f"Neuron is part of tracklet {name} with distance {dist}")

        if dist < max_dist:
            df_single_track = df_tracklets[name]
            all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track)
            viewer.add_tracks(track_of_point, name=name)

    napari.run()


if __name__ == "__main__":
    main()
