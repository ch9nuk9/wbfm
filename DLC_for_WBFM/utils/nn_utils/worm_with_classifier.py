from dataclasses import dataclass

from DLC_for_WBFM.utils.nn_utils.model_image_classifier import NeuronEmbeddingModel


PATH_TO_MODEL=""


@dataclass
def WormWithClassifier():
    model: NeuronEmbeddingModel
    all_frames: list = None

    t_template: int = None

    def setup(self):
        pass