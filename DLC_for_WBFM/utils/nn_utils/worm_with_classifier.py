import os.path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from DLC_for_WBFM.utils.neuron_matching.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.nn_utils.model_image_classifier import NeuronEmbeddingModel

PATH_TO_MODEL = "/scratch/neurobiology/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/nn_checkpoints/classifier_36_neurons.ckpt"
if not os.path.exists(PATH_TO_MODEL):
    raise FileNotFoundError(PATH_TO_MODEL)

# TODO: also save hyperparameters (doesn't work in jupyter notebooks)
hparams = dict(num_classes=36)


@dataclass
class WormWithNeuronClassifier:
    """Tracks neurons using a feature-space embedding and pre-calculated Frame objects"""
    template_frame: ReferenceFrame

    model: NeuronEmbeddingModel = None
    path_to_model: str = None

    embedding_template: torch.tensor = None
    labels_template: list = None

    def __post_init__(self):
        if self.path_to_model is None:
            self.path_to_model = PATH_TO_MODEL

        self.model = NeuronEmbeddingModel.load_from_checkpoint(checkpoint_path=self.path_to_model,
                                                               **hparams)
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
            # err
            matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
            matches = np.array(matches)
            conf = np.array([np.tanh(conf_matrix[i0, i1]) for i0, i1 in matches])
            matches_with_conf = [[m[0], m[1], c] for m, c in zip(matches, conf)]

        return matches_with_conf

    # def __repr__(self):
    #     return f"Worm Tracker based on network: {self.path_to_model}"
