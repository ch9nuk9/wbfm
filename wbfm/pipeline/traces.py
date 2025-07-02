import os
from collections import defaultdict

import numpy as np

from wbfm.utils import traces
from wbfm.utils.external.utils_zarr import zip_raw_data_zarr
from wbfm.utils.general.postprocessing.utils_metadata import region_props_all_volumes, \
    _convert_nested_dict_to_dataframe
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile, ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.traces.traces_pipeline import _unpack_configs_for_traces, match_segmentation_and_tracks, \
    _unpack_configs_for_extraction, _save_traces_as_hdf_and_update_configs
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.plot_traces import make_default_summary_plots_using_config
from wbfm.utils.visualization.utils_segmentation import _unpack_config_reindexing, reindex_segmentation


def match_segmentation_and_tracks_using_config(project_data: ProjectData,
                                               allow_only_global_tracker: bool = False,
                                               match_using_indices: bool = False,
                                               DEBUG: bool = False) -> None:
    """
    Connect the 3d traces to previously segmented masks

    NOTE: This assumes that the positions of the global tracks may be non-trivially different
    e.g. from a different tracking algorithm

    Get both red and green traces for each neuron


    Parameters
    ----------
    segment_cfg
    track_cfg
    traces_cfg
    project_cfg
    allow_only_global_tracker
    match_using_indices - If True (default=False), match using indices instead of xyz coordinates
    DEBUG

    Returns
    -------

    """
    project_cfg = project_data.project_config
    track_cfg = project_cfg.get_tracking_config()
    traces_cfg = project_cfg.get_traces_config()
    max_dist, params_start_volume, num_frames = _unpack_configs_for_traces(project_cfg, track_cfg)
    final_tracks = project_data.final_tracks

    # Sanity check: make sure that this is not just the global tracker, unless that is explicitly allowed
    if not allow_only_global_tracker:
        # Load the file, which saves which filename was used for the global tracks
        _ = project_data.intermediate_global_tracks
        if project_data.final_tracks_fname == project_data.intermediate_tracks_fname:
            raise ValueError("Final tracks and global tracks are the same. "
                             "If you want to use the global tracker, set allow_only_global_tracker=True")

    # Match -> Reindex raw segmentation -> Get traces
    final_neuron_names = get_names_from_df(final_tracks)
    for name in final_neuron_names:
        assert 'tracklet' not in name, f"Improper name found: {name}"

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask
    # Initialize multi-index dataframe for data
    all_matches = defaultdict(list)  # key = i_vol; val = Nx3-element list
    # TODO: Why is this one frame too short?
    frame_list = list(range(params_start_volume, num_frames + params_start_volume - 1))

    if not match_using_indices:
        coords = ['z', 'x', 'y']
        def _get_zxy_from_pandas(t):
            all_zxy = np.zeros((len(final_neuron_names), 3))
            for i, name in enumerate(final_neuron_names):
                all_zxy[i, :] = np.asarray(final_tracks[name][coords].loc[t])
            return all_zxy

        project_cfg.logger.info("Matching segmentation and tracked positions...")
        if DEBUG:
            frame_list = frame_list[:2]  # Shorten (to avoid break)
        match_segmentation_and_tracks(_get_zxy_from_pandas, all_matches, frame_list, max_dist,
                                      project_data, DEBUG=DEBUG)
    else:
        raise NotImplementedError("Matching using indices is not yet implemented")

    relative_fname = traces_cfg.config['all_matches']
    project_cfg.pickle_data_in_local_project(all_matches, relative_fname)


def extract_traces_using_config(project_data: ProjectData,
                                name_mode='neuron',
                                DEBUG=False):
    """
    Final step that loops through original data and extracts traces using labeled masks
    """
    project_cfg = project_data.project_config
    traces_cfg = project_cfg.get_traces_config()
    coords, reindexed_masks, frame_list, params_start_volume = \
        _unpack_configs_for_extraction(project_cfg, traces_cfg)

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


def calc_paper_traces_using_config(project_data: ProjectData,
                                   DEBUG=False):
    """
    Calculate paper traces using the traces config
    """
    # Also produce the paper-style "final" traces, and copy them to the final traces folder
    project_data.calc_all_paper_traces()
    project_data.copy_paper_traces_to_main_folder()

def reindex_segmentation_using_config(project_data: ProjectData, DEBUG=False):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    project_cfg = project_data.project_config
    traces_cfg = project_cfg.get_traces_config()
    segment_cfg = project_cfg.get_segmentations_config()
    all_matches, raw_seg_masks, new_masks, min_confidence, out_fname = _unpack_config_reindexing(traces_cfg, segment_cfg, project_cfg)
    reindex_segmentation(DEBUG, all_matches, raw_seg_masks, new_masks, min_confidence)

    return out_fname


def full_step_4_make_traces_from_config(project_cfg, allow_only_global_tracker=False, 
                                        DEBUG=False, **project_kwargs):
    project_dir = project_cfg.project_dir
    project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
    # Set environment variables to (try to) deal with rare blosc decompression errors
    os.environ["BLOSC_NOLOCK"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"
    with safe_cd(project_dir):
        # Overwrites matching pickle object; nothing needs to be reloaded
        match_segmentation_and_tracks_using_config(project_data,
                                                   allow_only_global_tracker=allow_only_global_tracker,
                                                   DEBUG=DEBUG)

        # Creates segmentations indexed to tracking
        project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
        new_mask_fname = reindex_segmentation_using_config(project_data)

        # Zips the reindexed segmentations to shrink requirements
        out_fname_zip = str(zip_raw_data_zarr(new_mask_fname))
        traces_cfg = project_data.project_config.get_traces_config()
        relative_fname = traces_cfg.unresolve_absolute_path(out_fname_zip)
        traces_cfg.config['reindexed_masks'] = relative_fname
        traces_cfg.update_self_on_disk()

        # Reads masks from disk, and writes traces
        project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
        extract_traces_using_config(project_data, name_mode='neuron', DEBUG=DEBUG)

        # Also produce the paper-style "final" traces, and copy them to the final traces folder
        project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
        calc_paper_traces_using_config(project_data)

        # By default make some visualizations
        project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
        make_default_summary_plots_using_config(project_data)
