import concurrent
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_candidates
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from fDNC.src.DNC_predict import pre_matt, predict_matches, filter_matches
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, ConfigFileWithProjectContext
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name
from DLC_for_WBFM.utils.xinwei_fdnc.formatting import zimmer2leifer

default_package_path = "/scratch/zimmer/Charles/github_repos/fDNC_Neuron_ID"


def load_prediction_options(custom_template=None, path_to_folder=None):
    if path_to_folder is None:
        path_to_folder = default_package_path

    model_path = os.path.join(path_to_folder, 'model', 'model.bin')
    prediction_options = dict(
        cuda=False,
        model_path=model_path
    )
    if custom_template is None:
        temp_fname = os.path.join(default_package_path, 'Data', 'Example', 'template.data')
        temp = pre_matt(temp_fname)
        template = temp['pts']
        template_label = temp['name']
    else:
        template = custom_template
        template_label = None

    return prediction_options, template, template_label


def track_using_fdnc(project_data: ProjectData,
                     prediction_options,
                     template,
                     match_confidence_threshold,
                     full_video_not_training=True):
    if full_video_not_training:
        num_frames = project_data.num_frames

        def get_pts(i):
            these_pts = project_data.get_centroids_as_numpy(i)
            return zimmer2leifer(these_pts)
    else:
        num_frames = project_data.reindexed_metadata_training.num_frames

        def get_pts(i):
            these_pts = project_data.get_centroids_as_numpy_training(i)
            return zimmer2leifer(these_pts)

    # def _parallel_func(i_frame):
    #
    #     pts = project_dat.get_centroids_as_numpy(i_frame)
    #     pts_scaled = zimmer2leifer(pts)
    #     # Match
    #     matches = predict_matches(test_pos=pts_scaled, **prediction_options)
    #     matches = filter_matches(matches, match_confidence_threshold)
    #
    #     # For each match, save location
    #     for m in matches:
    #         this_unscaled_pt = pts[m[1]]
    #         this_template_idx = m[0]
    #
    #         neuron_arrays[this_template_idx][i_frame, :3] = this_unscaled_pt
    #         neuron_arrays[this_template_idx][i_frame, 3] = m[2]  # Match confidence

    # Main loop
    # with tqdm(total=len(num_frames)) as pbar:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #         futures = {executor.submit(_parallel_func, i): i for i in range(num_frames)}
    #         for future in concurrent.futures.as_completed(futures):
    #             _ = future.result()
    #             pbar.update(1)

    all_matches = []
    for i_frame in tqdm(range(num_frames), total=num_frames, leave=False):
        pts_scaled = get_pts(i_frame)
        matches = predict_matches(test_pos=pts_scaled, template_pos=template, **prediction_options)
        matches = filter_matches(matches, match_confidence_threshold)

        all_matches.append(matches)

    return all_matches


def template_matches_to_dataframe(project_data: ProjectData,
                                  all_matches):
    num_frames = project_data.num_frames
    coords = ['z', 'x', 'y', 'likelihood']
    sz = (num_frames, len(coords))
    neuron_arrays = defaultdict(lambda: np.zeros(sz))

    for i_frame, these_matches in enumerate(tqdm(all_matches, leave=False)):
        pts = project_data.get_centroids_as_numpy(i_frame)
        # For each match, save location
        for m in these_matches:
            this_unscaled_pt = pts[m[1]]
            this_template_idx = m[0]

            neuron_arrays[this_template_idx][i_frame, :3] = this_unscaled_pt
            neuron_arrays[this_template_idx][i_frame, 3] = m[2]  # Match confidence

    # Convert to pandas multiindexing formatting
    new_dict = {}
    for i_template, data in neuron_arrays.items():
        for i_col, coord_name in enumerate(coords):
            # Note: these neuron names are final for all subsequent steps
            k = (int2name(i_template + 1), coord_name)
            new_dict[k] = data[:, i_col]

    df = pd.DataFrame(new_dict)

    return df


def generate_templates_from_training_data(project_data: ProjectData):
    all_templates = []
    num_templates = project_data.reindexed_metadata_training.num_frames

    for i in range(num_templates):
        custom_template = project_data.get_centroids_as_numpy_training(i)
        all_templates.append(zimmer2leifer(custom_template))
    return all_templates


def track_using_fdnc_multiple_templates(project_data: ProjectData,
                                        base_prediction_options,
                                        match_confidence_threshold):
    all_templates = generate_templates_from_training_data(project_data)

    def _parallel_func(template):
        return track_using_fdnc(project_data, base_prediction_options, template, match_confidence_threshold)

    max_workers = round(project_data.reindexed_metadata_training.num_frames / 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_jobs = [executor.submit(_parallel_func, template) for template in all_templates]
        matches_per_template = [job.result() for job in submitted_jobs]

    # Combine the matches between each frame and template
    final_matches = combine_multiple_template_matches(matches_per_template)

    return final_matches


def combine_multiple_template_matches(matches_per_template):
    final_matches = []
    num_frames = len(matches_per_template[0])
    num_templates = len(matches_per_template)
    for i_frame in range(num_frames):
        candidate_matches = []
        for i_template in range(num_templates):
            candidate_matches.extend(matches_per_template[i_template][i_frame])
        # Reduce individual confidences so they are an average, not a sum
        candidate_matches = [(m[0], m[1], m[2] / num_templates) for m in candidate_matches]

        matches, conf, _ = calc_bipartite_from_candidates(candidate_matches, min_conf=0.1)
        match_and_conf = [(m[0], m[1], c) for m, c in zip(matches, conf)]
        final_matches.append(match_and_conf)
    return final_matches


def track_using_fdnc_from_config(project_cfg: ModularProjectConfig,
                                 tracks_cfg: ConfigFileWithProjectContext,
                                 DEBUG=False):
    match_confidence_threshold, prediction_options, template, project_data, use_multiple_templates = \
        _unpack_for_fdnc(project_cfg, tracks_cfg, DEBUG)

    if use_multiple_templates:
        logging.info("Tracking using multiple templates")
        all_matches = track_using_fdnc_multiple_templates(project_data, prediction_options, match_confidence_threshold)
    else:
        logging.info("Tracking using single template")
        all_matches = track_using_fdnc(project_data, prediction_options, template, match_confidence_threshold)

    logging.info("Converting matches to dataframe format")
    df = template_matches_to_dataframe(project_data, all_matches)

    logging.info("Saving tracks and matches")
    with safe_cd(project_cfg.project_dir):
        _save_tracks_and_matches(all_matches, df, project_cfg, tracks_cfg)


def _save_tracks_and_matches(all_matches, df, project_cfg, tracks_cfg):
    output_df_fname = tracks_cfg.config['leifer_params']['output_df_fname']
    df.to_hdf(output_df_fname, key='df_with_missing')

    tracks_cfg.config['final_3d_tracks_df'] = str(output_df_fname)
    tracks_cfg.update_on_disk()
    # For later visualization
    output_df_fname = Path(output_df_fname).with_suffix('.csv')
    df.to_csv(str(output_df_fname))
    output_pickle_fname = Path(output_df_fname).with_name('fdnc_matches.pickle')
    project_cfg.pickle_in_local_project(all_matches, output_pickle_fname)


def _unpack_for_fdnc(project_cfg, tracks_cfg, DEBUG):
    use_zimmer_template = tracks_cfg.config['leifer_params']['use_zimmer_template']
    use_multiple_templates = tracks_cfg.config['leifer_params']['use_multiple_templates']
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    if DEBUG:
        project_data.project_config.config['dataset_params']['num_frames'] = 2
    if use_zimmer_template:
        # TODO: use a hand-curated segmentation
        custom_template = project_data.get_centroids_as_numpy(0)
        custom_template = zimmer2leifer(custom_template)
    else:
        custom_template = None
    prediction_options, template, _ = load_prediction_options(custom_template=custom_template)
    match_confidence_threshold = tracks_cfg.config['leifer_params']['match_confidence_threshold']
    return match_confidence_threshold, prediction_options, template, project_data, use_multiple_templates
