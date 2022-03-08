import os.path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.neuron_matching.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.nn_utils.model_image_classifier import NeuronEmbeddingModel
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData, template_matches_to_dataframe

PATH_TO_MODEL = "/scratch/neurobiology/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/nn_checkpoints/classifier_36_neurons.ckpt"
if not os.path.exists(PATH_TO_MODEL):
    raise FileNotFoundError(PATH_TO_MODEL)

# TODO: also save hyperparameters (doesn't work in jupyter notebooks)
HPARAMS = dict(num_classes=36)


@dataclass
class WormWithNeuronClassifier:
    """Tracks neurons using a feature-space embedding and pre-calculated Frame objects"""
    template_frame: ReferenceFrame

    model_type: callable = NeuronEmbeddingModel
    model: NeuronEmbeddingModel = None
    path_to_model: str = None
    hparams: dict = None

    embedding_template: torch.tensor = None
    labels_template: list = None

    def __post_init__(self):
        if self.path_to_model is None:
            self.path_to_model = PATH_TO_MODEL
        if self.hparams is None:
            self.hparams = HPARAMS

        self.model = self.model_type.load_from_checkpoint(checkpoint_path=self.path_to_model, **self.hparams)
        # self.model

        self.initialize_template()

    def initialize_template(self, template_frame: ReferenceFrame = None):
        if template_frame is None:
            template_frame = self.template_frame
        else:
            self.template_frame = template_frame
        if template_frame is None:
            raise NotImplementedError("Must pass template_frame or initialize self.t_template")

        features = torch.from_numpy(template_frame.all_features)
        self.embedding_template = self.model.embed(features.to(self.model.device))
        # TODO: better naming?
        self.labels_template = list(range(features.shape[0]))

    def match_target_frame(self, target_frame: ReferenceFrame):

        with torch.no_grad():
            query_features = torch.tensor(target_frame.all_features).to(self.model.device)
            query_embedding = self.model.embed(query_features)

            distances = torch.cdist(self.embedding_template, query_embedding)
            conf_matrix = torch.nan_to_num(torch.softmax(1.0 / distances, dim=0), nan=1.0)

            matches = linear_sum_assignment(conf_matrix, maximize=True)
            matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
            matches = np.array(matches)
            conf = np.array([np.tanh(conf_matrix[i0, i1]) for i0, i1 in matches])
            matches_with_conf = [[m[0], m[1], c] for m, c in zip(matches, conf)]

        return matches_with_conf

    def __repr__(self):
        return f"Worm Tracker based on network: {self.path_to_model}"


def track_using_embedding_from_config(project_cfg, DEBUG):
    project_data = ProjectData.load_final_project_data_from_config(project_cfg,
                                                                   to_load_tracklets=True, to_load_frames=True)

    tracking_cfg = project_data.project_config.get_tracking_config()
    t_template = tracking_cfg.config['final_3d_tracks']['template_time_point']

    num_frames = project_data.num_frames
    if DEBUG:
        num_frames = 3
    all_frames = project_data.raw_frames
    tracker = WormWithNeuronClassifier(template_frame=all_frames[t_template])

    all_matches = []
    for t in tqdm(range(num_frames)):
        matches_with_conf = tracker.match_target_frame(all_frames[t])

        all_matches.append(matches_with_conf)

    df = template_matches_to_dataframe(project_data, all_matches)

    # Save
    out_fname = '3-tracking/postprocessing/df_tracks_embedding.h5'
    tracking_cfg.h5_in_local_project(df, out_fname, also_save_csv=True)
    tracking_cfg.config['leifer_params']['output_df_fname'] = out_fname
    tracking_cfg.update_on_disk()
