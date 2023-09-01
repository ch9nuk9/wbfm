##
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from wbfm.barlow_project.utils import NeuronImageFeaturesDataModuleFromMultipleProjects, \
    load_data_with_ground_truth
from wbfm.utils.nn_utils.superglue import SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject, \
    SuperGlueModel
from wbfm.utils.nn_utils.worm_with_classifier import PATH_TO_SUPERGLUE_MODEL


def main():
    # Load data
    all_project_data = load_data_with_ground_truth()

    # Set up the network
    base_dataset_class = SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject
    batch_size = 1
    max_epochs = 20
    # Take the last gpu; for now, multiple gpus are not supported
    gpus = [torch.cuda.device_count() - 1]
    train_loader = NeuronImageFeaturesDataModuleFromMultipleProjects(batch_size=batch_size,
                                                                     all_project_data=all_project_data,
                                                                     base_dataset_class=base_dataset_class,
                                                                     dataset_kwargs=dict(
                                                                         num_to_calculate=1000))  # Calculates from each project
    # Explicitly setup to see if there are problems
    train_loader.setup()
    # Start from pretrained
    model = SuperGlueModel.load_from_checkpoint(PATH_TO_SUPERGLUE_MODEL)
    # model = SuperGlueModel(feature_dim=840, lr=1e-5)
    model.lr = 1e-5
    with wandb.init(project="superglue_training_multiple_projects_fixed_r2w4", entity="charlesfieseler") as run:
        wandb_logger = WandbLogger()

        trainer = Trainer(gpus=gpus, max_epochs=max_epochs, terminate_on_nan=True,
                          stochastic_weight_avg=True,
                          logger=wandb_logger)
        wandb_logger.watch(model, log='all', log_freq=1)

        trainer.fit(model, train_loader)
    out_folder = '/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/nn_checkpoints'
    model_fname = os.path.join(out_folder, 'superglue_neurons_4_datasets_06_22.ckpt')
    trainer.save_checkpoint(model_fname)


if __name__ == "__main__":
    main()
