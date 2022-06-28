import logging
import os
from dataclasses import dataclass
from typing import Union, List
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.gui.utils.utils_gui import change_viewer_time_point
from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePair
import napari

from DLC_for_WBFM.utils.projects.utils_filenames import get_sequential_filename
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_tracks_from_match_list, napari_labels_from_frames
from DLC_for_WBFM.utils.visualization.napari_utils import napari_labels_from_traces_dataframe, NapariPropertyHeatMapper


@dataclass
class NapariLayerInitializer:

    @staticmethod
    def napari_of_single_match(project_data,
                               pair,
                               which_matches='final_matches',
                               this_match: FramePair = None,
                               rigidly_align_volumetric_images=False,
                               min_confidence=0.0) -> napari.Viewer:
        if np.isscalar(pair):
            pair = (pair, pair+1)

        if this_match is None:
            this_match: FramePair = project_data.raw_matches[pair]

        t0, t1 = pair
        dat0, dat1 = project_data.red_data[t0, ...], project_data.red_data[t1, ...]
        seg0, seg1 = project_data.raw_segmentation[t0, ...], project_data.raw_segmentation[t1, ...]
        this_match.load_raw_data(dat0, dat1)
        if rigidly_align_volumetric_images:
            # Ensure that both point cloud and data have rotations
            this_match.preprocess_data(force_rotation=True)
            # Load the rotated versions
            n0_zxy = this_match.pts0_preprocessed  # May be rotated
            dat0 = this_match.dat0_preprocessed
        else:
            # Keep the non-rotated versions
            n0_zxy = this_match.pts0

        n1_zxy = this_match.pts1
        raw_red_data = np.stack([dat0, dat1])
        raw_seg_data = np.stack([seg0, seg1])
        # Scale to physical units
        z_to_xy_ratio = 1
        # z_to_xy_ratio = project_data.physical_unit_conversion.z_to_xy_ratio
        # n0_zxy[0, :] = z_to_xy_ratio * n0_zxy[0, :]
        # n1_zxy[0, :] = z_to_xy_ratio * n1_zxy[0, :]

        list_of_matches = getattr(this_match, which_matches)
        list_of_matches = [m for m in list_of_matches if -1 not in m]
        list_of_matches = [m for m in list_of_matches if m[2] > min_confidence]

        all_tracks_list = napari_tracks_from_match_list(list_of_matches, n0_zxy, n1_zxy)

        v = napari.view_image(raw_red_data, ndisplay=3, scale=(1.0, z_to_xy_ratio, 1.0, 1.0))
        v.add_labels(raw_seg_data, scale=(1.0, z_to_xy_ratio, 1.0, 1.0), visible=False)

        # This should not remember the original time point
        df = project_data.final_tracks.loc[[t0], :].set_index(pd.Index([0]))
        options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
        options['name'] = 'n0_final_id'
        options['n_dimensional'] = True
        v.add_points(**options)

        # This should not remember the original time point
        df = project_data.final_tracks.loc[[t1], :].set_index(pd.Index([0]))
        options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
        options['name'] = 'n1_final_id'
        options['text']['color'] = 'green'
        options['n_dimensional'] = True
        options['symbol'] = 'x'
        v.add_points(**options)
        # v.add_points(n0_zxy, size=3, face_color='green', symbol='x', n_dimensional=True)
        # v.add_points(n1_zxy, size=3, face_color='blue', symbol='o', n_dimensional=True)
        v.add_tracks(all_tracks_list, head_length=2, name=which_matches)

        # Add text overlay; temporarily change the neuron locations on the frame
        original_zxy = this_match.frame0.neuron_locs
        this_match.frame0.neuron_locs = n0_zxy
        frames = {0: this_match.frame0, 1: this_match.frame1}
        options = napari_labels_from_frames(frames, num_frames=2, to_flip_zxy=False)
        options['name'] = "Neuron ID in list"
        v.add_points(**options)
        this_match.frame0.neuron_locs = original_zxy

        return v

    @staticmethod
    def add_layers_to_viewer(project_data, viewer=None, which_layers: Union[str, List[str]] = 'all',
                             to_remove_flyback=False, check_if_layers_exist=False,
                             dask_for_segmentation=True, force_all_visible=False):
        if viewer is None:
            viewer = napari.Viewer(ndisplay=3)
        if which_layers == 'all':
            which_layers = ['Red data', 'Green data', 'Raw segmentation', 'Colored segmentation',
                            'Neuron IDs', 'Intermediate global IDs']
        if check_if_layers_exist:
            # NOTE: only works if the layer names are the same as these convinience names
            new_layers = set(which_layers) - set([layer.name for layer in viewer.layers])
            which_layers = list(new_layers)

        project_data.logger.info(f"Finished loading data, adding following layers: {which_layers}")
        z_to_xy_ratio = project_data.physical_unit_conversion.z_to_xy_ratio
        if to_remove_flyback:
            clipping_list = [{'position': [2*z_to_xy_ratio, 0, 0], 'normal': [1, 0, 0], 'enabled': True}]
        else:
            clipping_list = []

        if 'Red data' in which_layers:
            viewer.add_image(project_data.red_data, name="Red data", opacity=0.5, colormap='PiYG',
                             contrast_limits=[0, 200],
                             scale=(1.0, z_to_xy_ratio, 1.0, 1.0),
                             experimental_clipping_planes=clipping_list)
        if 'Green data' in which_layers:
            visibility = force_all_visible
            viewer.add_image(project_data.green_data, name="Green data", opacity=0.5, colormap='green',
                             visible=visibility,
                             contrast_limits=[0, 200],
                             scale=(1.0, z_to_xy_ratio, 1.0, 1.0),
                             experimental_clipping_planes=clipping_list)
        if 'Raw segmentation' in which_layers:
            visibility = force_all_visible
            seg_array = project_data.raw_segmentation
            viewer.add_labels(seg_array, name="Raw segmentation",
                              scale=(1.0, z_to_xy_ratio, 1.0, 1.0), opacity=0.8, visible=visibility,
                              rendering='translucent')
        if 'Colored segmentation' in which_layers and project_data.segmentation is not None:
            visibility = force_all_visible
            viewer.add_labels(project_data.segmentation, name="Colored segmentation",
                              scale=(1.0, z_to_xy_ratio, 1.0, 1.0), opacity=0.4, visible=force_all_visible)

        # Add a text overlay
        if 'Neuron IDs' in which_layers:
            df = project_data.red_traces
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
            options['visible'] = force_all_visible
            viewer.add_points(**options)

        if 'GT IDs' in which_layers:
            # Not added by default!
            df = project_data.final_tracks
            neurons_that_are_finished = project_data.finished_neuron_names
            neuron_name_dict = {name: f"GT_{name.split('_')[1]}" for name in neurons_that_are_finished}
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio,
                                                          neuron_name_dict=neuron_name_dict)
            options['name'] = 'GT IDs'
            options['text']['color'] = 'red'
            options['visible'] = force_all_visible
            viewer.add_points(**options)

        if 'Intermediate global IDs' in which_layers and project_data.intermediate_global_tracks is not None:
            df = project_data.intermediate_global_tracks
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
            options['name'] = 'Intermediate global IDs'
            options['text']['color'] = 'green'
            options['visible'] = force_all_visible
            viewer.add_points(**options)

        # Special layers from the heatmapper class
        heat_mapper = NapariPropertyHeatMapper(project_data.red_traces, project_data.green_traces)
        for layer_tuple in which_layers:
            if not isinstance(layer_tuple, tuple):
                continue
            elif 'heatmap' not in layer_tuple:
                logging.warning(f"Skipping tuple: {layer_tuple}")
                continue
            else:
                layer_name = layer_tuple[1]

            prop_dict = getattr(heat_mapper, layer_name)()
            # Note: this layer must be visible for the prop_dict to work correctly
            _layer = viewer.add_labels(project_data.segmentation, name=layer_name,
                                       scale=(1.0, z_to_xy_ratio, 1.0, 1.0),
                                       opacity=0.4, visible=True, rendering='translucent')
            _layer.color = prop_dict
            _layer.color_mode = 'direct'

        project_data.logger.info(f"Finished adding layers {which_layers}")

        return viewer


def take_screenshot_using_project(project_data, additional_layers, base_layers=None, t_target=None, **kwargs):
    if t_target is None:
        tracking_cfg = project_data.project_config.get_tracking_config()
        t_target = tracking_cfg.config['final_3d_tracks']['template_time_point']
    if base_layers is None:
        base_layers = ['Red data']

    viewer = NapariLayerInitializer().add_layers_to_viewer(project_data, which_layers=base_layers,
                                                           force_all_visible=True, **kwargs)
    change_viewer_time_point(viewer, t_target=t_target)
    for layer in tqdm(additional_layers):
        if not isinstance(layer, list):
            layer = [layer]
        NapariLayerInitializer().add_layers_to_viewer(project_data, viewer=viewer, which_layers=layer,
                                                      force_all_visible=True, **kwargs)

        # For the output name, assume I'm only adding one layer type over the base layer
        output_folder = project_data.project_config.get_visualization_dir()
        layer_name = layer[0]
        if isinstance(layer_name, tuple):
            layer_name = layer_name[1]
        fname = os.path.join(output_folder, f'{layer_name}.png')
        fname = get_sequential_filename(fname)
        viewer.screenshot(path=fname)

        viewer.layers.remove(layer_name)
