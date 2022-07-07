from collections import defaultdict

import numpy as np

from wbfm.utils.general.postprocessing.utils_metadata import region_props_all_volumes, \
    _convert_nested_dict_to_dataframe
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile, ModularProjectConfig
from wbfm.utils.traces.traces_pipeline import _unpack_configs_for_traces, match_segmentation_and_tracks, \
    _unpack_configs_for_extraction, _save_traces_as_hdf_and_update_configs
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.utils_segmentation import _unpack_config_reindexing, reindex_segmentation


def match_segmentation_and_tracks_using_config(segment_cfg: SubfolderConfigFile,
                                               track_cfg: SubfolderConfigFile,
                                               traces_cfg: SubfolderConfigFile,
                                               project_cfg: ModularProjectConfig,
                                               DEBUG: bool = False) -> None:
    """
    Connect the 3d traces to previously segmented masks

    NOTE: This assumes that the global tracks may be non-trivially different, e.g. from a different tracking algorithm

    Get both red and green traces for each neuron
    """
    final_tracks, green_fname, red_fname, max_dist, num_frames, params_start_volume = _unpack_configs_for_traces(
        project_cfg, segment_cfg, track_cfg)

    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Match -> Reindex raw segmentation -> Get traces
    final_neuron_names = get_names_from_df(final_tracks)
    for name in final_neuron_names:
        assert 'tracklet' not in name, f"Improper name found: {name}"
    coords = ['z', 'x', 'y']

    def _get_zxy_from_pandas(t):
        all_zxy = np.zeros((len(final_neuron_names), 3))
        for i, name in enumerate(final_neuron_names):
            all_zxy[i, :] = np.asarray(final_tracks[name][coords].loc[t])
        return all_zxy

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask
    # Initialize multi-index dataframe for data
    # TODO: Why is this one frame too short?
    frame_list = list(range(params_start_volume, num_frames + params_start_volume - 1))
    all_matches = defaultdict(list)  # key = i_vol; val = Nx3-element list
    project_cfg.logger.info("Matching segmentation and tracked positions...")
    if DEBUG:
        frame_list = frame_list[:2]  # Shorten (to avoid break)
    match_segmentation_and_tracks(_get_zxy_from_pandas, all_matches, frame_list, max_dist,
                                  project_data, DEBUG=DEBUG)

    relative_fname = traces_cfg.config['all_matches']
    project_cfg.pickle_data_in_local_project(all_matches, relative_fname)


def extract_traces_using_config(project_cfg: SubfolderConfigFile,
                                traces_cfg: SubfolderConfigFile,
                                name_mode='neuron',
                                DEBUG=False):
    """
    Final step that loops through original data and extracts traces using labeled masks
    """
    coords, reindexed_masks, frame_list, params_start_volume = \
        _unpack_configs_for_extraction(project_cfg, traces_cfg)
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    red_all_neurons, green_all_neurons = region_props_all_volumes(
        reindexed_masks,
        project_data.red_data,
        project_data.green_data,
        frame_list,
        params_start_volume,
        name_mode
    )

    df_green = _convert_nested_dict_to_dataframe(coords, frame_list, green_all_neurons)
    df_red = _convert_nested_dict_to_dataframe(coords, frame_list, red_all_neurons)

    final_neuron_names = get_names_from_df(df_red)

    _save_traces_as_hdf_and_update_configs(final_neuron_names, df_green, df_red, traces_cfg)


def reindex_segmentation_using_config(traces_cfg: SubfolderConfigFile,
                                      segment_cfg: SubfolderConfigFile,
                                      project_cfg: ModularProjectConfig,
                                      DEBUG=False):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    all_matches, raw_seg_masks, new_masks, min_confidence, out_fname = _unpack_config_reindexing(traces_cfg, segment_cfg, project_cfg)
    reindex_segmentation(DEBUG, all_matches, raw_seg_masks, new_masks, min_confidence)

    return out_fname
