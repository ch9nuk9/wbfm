import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import fill_missing_indices_with_nan
from DLC_for_WBFM.utils.neuron_matching.long_range_matching import _unpack_for_track_tracklet_matching, \
    extend_tracks_using_global_tracking, bipartite_matching_on_each_time_slice, greedy_matching_using_node_class, \
    combine_tracklets_using_matching, _save_graphs_and_combined_tracks

from DLC_for_WBFM.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching, \
    combine_dataframes_using_bipartite_matching
from DLC_for_WBFM.utils.nn_utils.superglue import SuperGlueUnpacker
from DLC_for_WBFM.utils.nn_utils.worm_with_classifier import _unpack_project_for_global_tracking, \
    WormWithSuperGlueClassifier, track_using_template, generate_random_template_times, WormWithNeuronClassifier
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.tracklets.tracklet_class import TrackedWorm, DetectedTrackletsAndNeurons
from DLC_for_WBFM.utils.tracklets.utils_tracklets import split_all_tracklets_at_once


def track_using_superglue_using_config(project_cfg, DEBUG):
    all_frames, num_frames, num_random_templates, project_data, t_template, tracking_cfg, use_multiple_templates = _unpack_project_for_global_tracking(
        DEBUG, project_cfg)
    superglue_unpacker = SuperGlueUnpacker(project_data=project_data, t_template=t_template)
    tracker = WormWithSuperGlueClassifier(superglue_unpacker=superglue_unpacker)

    if not use_multiple_templates:
        df_final = track_using_template(all_frames, num_frames, project_data, tracker)
    else:
        all_templates = generate_random_template_times(num_frames, num_random_templates, t_template)
        project_cfg.logger.info(f"Using {num_random_templates} templates at t={all_templates}")
        # All subsequent dataframes will have their names mapped to this
        df_base = track_using_template(all_frames, num_frames, project_data, tracker)
        all_dfs = [df_base]
        for i, t in enumerate(tqdm(all_templates[1:])):
            superglue_unpacker = SuperGlueUnpacker(project_data=project_data, t_template=t)
            tracker = WormWithSuperGlueClassifier(superglue_unpacker=superglue_unpacker)
            df = track_using_template(all_frames, num_frames, project_data, tracker)
            df, _, _, _ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)
            all_dfs.append(df)

        tracking_cfg.config['t_templates'] = all_templates
        df_final = combine_dataframes_using_bipartite_matching(all_dfs)

    # Save
    out_fname = '3-tracking/postprocessing/df_tracks_superglue.h5'
    tracking_cfg.h5_data_in_local_project(df_final, out_fname, also_save_csv=True)
    tracking_cfg.config['leifer_params']['output_df_fname'] = out_fname

    tracking_cfg.update_self_on_disk()


def track_using_embedding_using_config(project_cfg, DEBUG):
    all_frames, num_frames, num_random_templates, project_data, t_template, tracking_cfg, use_multiple_templates = _unpack_project_for_global_tracking(
        DEBUG, project_cfg)

    if not use_multiple_templates:
        tracker = WormWithNeuronClassifier(template_frame=all_frames[t_template])
        df_final = track_using_template(all_frames, num_frames, project_data, tracker)
    else:
        all_templates = generate_random_template_times(num_frames, num_random_templates, t_template)
        # All subsequent dataframes will have their names mapped to this
        t = all_templates[0]
        tracker = WormWithNeuronClassifier(template_frame=all_frames[t])
        df_base = track_using_template(all_frames, num_frames, project_data, tracker)
        all_dfs = [df_base]
        for i, t in enumerate(tqdm(all_templates[1:])):
            tracker = WormWithNeuronClassifier(template_frame=all_frames[t])
            df = track_using_template(all_frames, num_frames, project_data, tracker)
            df, _, _, _ = rename_columns_using_matching(df_base, df)
            all_dfs.append(df)

        tracking_cfg.config['t_templates'] = all_templates
        df_final = combine_dataframes_using_bipartite_matching(all_dfs)

    # Save
    out_fname = '3-tracking/postprocessing/df_tracks_embedding.h5'
    tracking_cfg.h5_data_in_local_project(df_final, out_fname, also_save_csv=True)
    tracking_cfg.config['leifer_params']['output_df_fname'] = out_fname

    tracking_cfg.update_self_on_disk()


def match_tracks_and_tracklets_using_config(project_config: ModularProjectConfig, to_save=True, verbose=0,
                                            auto_split_conflicts=True, DEBUG=False):
    """Replaces: final_tracks_from_tracklet_matches_from_config"""
    # Initialize project data and unpack
    logger = project_config.logger
    project_data = ProjectData.load_final_project_data_from_config(project_config, to_load_tracklets=True)
    df_global_tracks, min_confidence, min_overlap, num_neurons, only_use_previous_matches, outlier_threshold, \
    previous_matches, t_template, track_config, tracklets_and_neurons_class, use_multiple_templates, \
    use_previous_matches, tracklet_splitting_iterations = _unpack_for_track_tracklet_matching(project_data)

    # Add initial tracklets to neurons, then add matches (if any found before)
    logger.info(f"Initializing worm class with settings: \n"
                f"only_use_previous_matches={only_use_previous_matches}\n"
                f"use_previous_matches={use_previous_matches}\n"
                f"use_multiple_templates={use_multiple_templates}")

    def _initialize_worm(tracklets_obj, verbose=verbose):
        _worm_obj = TrackedWorm(detections=tracklets_obj, verbose=verbose)
        if only_use_previous_matches:
            _worm_obj.initialize_neurons_using_previous_matches(previous_matches)
        else:
            _worm_obj.initialize_neurons_at_time(t=t_template, num_expected_neurons=num_neurons,
                                                 df_global_tracks=df_global_tracks)
            if use_previous_matches:
                _worm_obj.add_previous_matches(previous_matches)
        # _worm_obj.initialize_all_neuron_tracklet_classifiers()
        if verbose >= 1:
            logger.info(f"Initialized worm object: {_worm_obj}")
        return _worm_obj

    worm_obj = _initialize_worm(tracklets_and_neurons_class)

    # Note: need to load this after the worm object is initialized, because the df may be modified
    df_tracklets = worm_obj.detections.df_tracklets_zxy
    df_tracklets_split = None

    extend_tracks_opt = dict(min_overlap=min_overlap, min_confidence=min_confidence,
                             outlier_threshold=outlier_threshold, verbose=verbose, DEBUG=DEBUG)
    if not only_use_previous_matches:
        logger.info("Adding all tracklet candidates to neurons")
        extend_tracks_using_global_tracking(df_global_tracks, df_tracklets, worm_obj, **extend_tracks_opt)

        # Build candidate graph, then postprocess it
        global_tracklet_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
        if not auto_split_conflicts:
            logger.info("Bipartite matching for each time slice subgraph")
            final_matching_with_conflict = bipartite_matching_on_each_time_slice(global_tracklet_neuron_graph,
                                                                                 df_tracklets)
            # Final step to remove time conflicts
            worm_obj.reinitialize_all_neurons_from_final_matching(final_matching_with_conflict)
            worm_obj.remove_conflicting_tracklets_from_all_neurons()
        else:
            # For metadata saving (the original worm with conflicts is otherwise not saved)
            logger.info("Calculating node matches for metadata purposes")
            final_matching_with_conflict = greedy_matching_using_node_class(global_tracklet_neuron_graph,
                                                                            node_class_to_match=1)

            logger.info("Iteratively splitting tracklets using track matching conflicts")
            for i_split in tqdm(range(tracklet_splitting_iterations)):
                split_list_dict = worm_obj.get_conflict_time_dictionary_for_all_neurons(
                    minimum_confidence=min_confidence)
                if len(split_list_dict) == 0:
                    logger.info(f"Found no further tracklet conflicts on iteration i={i_split}")
                    break
                else:
                    logger.info(f"Found conflicts on {len(split_list_dict)} tracklets")
                df_tracklets_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(df_tracklets, split_list_dict)
                tracklets_and_neurons_class2 = DetectedTrackletsAndNeurons(df_tracklets_split,
                                                                           project_data.segmentation_metadata,
                                                                           dataframe_output_filename=project_data.df_all_tracklets_fname)
                worm_obj2 = _initialize_worm(tracklets_and_neurons_class2, verbose=0)
                if i_split == tracklet_splitting_iterations:
                    # On the last iteration, allow short tracklets to be matched
                    extend_tracks_opt['min_overlap'] = 1
                conf2 = extend_tracks_using_global_tracking(df_global_tracks, df_tracklets_split, worm_obj2,
                                                            **extend_tracks_opt)
                # Overwrite original object, and continue
                worm_obj = worm_obj2
                df_tracklets = df_tracklets_split.astype(pd.SparseDtype("float", np.nan))

        # TODO: should I do this after tracklet-unique processing? For now the formats are a pain
        logger.info("Removing tracklets that have time conflicts on a single neuron ")
        worm_obj.remove_conflicting_tracklets_from_all_neurons()
        worm_obj.update_time_covering_ind_for_all_neurons()
    else:
        global_tracklet_neuron_graph = None
        final_matching_with_conflict = None

    no_conflict_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
    logger.info("Final matching to prevent the same tracklet assigned to multiple neurons")
    final_matching_no_conflict = greedy_matching_using_node_class(no_conflict_neuron_graph, node_class_to_match=1)
    df_new = combine_tracklets_using_matching(df_tracklets, final_matching_no_conflict)

    df_final, num_added = fill_missing_indices_with_nan(df_new)
    if num_added > 0:
        logger.warning(f"Some time points {num_added} are completely empty of tracklets, and are added as nan")

    # SAVE
    if to_save:
        with safe_cd(project_data.project_dir):
            _save_graphs_and_combined_tracks(df_final, final_matching_no_conflict, final_matching_with_conflict,
                                             global_tracklet_neuron_graph,
                                             track_config, worm_obj,
                                             df_tracklets_split)
    return df_final, final_matching_no_conflict, global_tracklet_neuron_graph, worm_obj
