import concurrent
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fDNC.src.DNC_predict import pre_matt, predict_matches, filter_matches
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.finished_project_data import finished_project_data
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
        prediction_options['template_pos'] = temp['pts']
    else:
        prediction_options['template_pos'] = custom_template

    return prediction_options


def track_using_fdnc(project_dat: finished_project_data,
                     prediction_options,
                     match_confidence_threshold):
    # Loop through detections and match all to template

    # Initialize
    num_frames = project_dat.num_frames
    coords = ['z', 'x', 'y', 'likelihood']

    sz = (num_frames, len(coords))
    neuron_arrays = defaultdict(lambda: np.zeros(sz))

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
    for i_frame in tqdm(range(num_frames), total=num_frames):

        pts = project_dat.get_centroids_as_numpy(i_frame)
        pts_scaled = zimmer2leifer(pts)
        # Match
        matches = predict_matches(test_pos=pts_scaled, **prediction_options)
        matches = filter_matches(matches, match_confidence_threshold)

        # For each match, save location
        for m in matches:
            this_unscaled_pt = pts[m[1]]
            this_template_idx = m[0]

            neuron_arrays[this_template_idx][i_frame, :3] = this_unscaled_pt
            neuron_arrays[this_template_idx][i_frame, 3] = m[2]  # Match confidence
        all_matches.append(matches)

    # Convert to pandas multiindexing formatting
    new_dict = {}
    for i_template, data in neuron_arrays.items():
        for i_col, coord_name in enumerate(coords):
            # Note: these neuron names are final for all subsequent steps
            k = (int2name(i_template + 1), coord_name)
            new_dict[k] = data[:, i_col]

    df = pd.DataFrame(new_dict)

    return df, all_matches


def track_using_fdnc_from_config(project_cfg: ModularProjectConfig,
                                 tracks_cfg: ConfigFileWithProjectContext):
    use_zimmer_template = tracks_cfg.config['leifer_params']['use_zimmer_template']
    project_dat = finished_project_data.load_final_project_data_from_config(project_cfg)
    if use_zimmer_template:
        # TODO: use a hand-curated segmentation
        custom_template = project_dat.get_centroids_as_numpy(0)
        custom_template = zimmer2leifer(custom_template)
    prediction_options = load_prediction_options(custom_template=custom_template)
    match_confidence_threshold = tracks_cfg.config['leifer_params']['match_confidence_threshold']
    output_df_fname = tracks_cfg.config['leifer_params']['output_df_fname']

    df, all_matches = track_using_fdnc(project_dat, prediction_options, match_confidence_threshold)

    # Save in main traces folder
    df.to_hdf(output_df_fname, key='df_with_missing')
    tracks_cfg.config['final_3d_tracks_df'] = str(output_df_fname)
    tracks_cfg.update_on_disk()

    # For visualization
    output_df_fname = Path(output_df_fname).with_suffix('.csv')
    df.to_csv(str(output_df_fname))

    output_pickle_fname = Path(output_df_fname).with_name('fdnc_matches.pickle')
    with open(output_pickle_fname, 'wb') as f:
        pickle.dump(all_matches, f)
