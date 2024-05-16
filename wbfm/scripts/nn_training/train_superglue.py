##
import os
from typing import List, Optional

import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import Dataset, random_split, DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from wbfm.utils.nn_utils.superglue import SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject, \
    SuperGlueModel
from wbfm.utils.nn_utils.worm_with_classifier import PATH_TO_SUPERGLUE_MODEL
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_redo_steps import correct_tracks_dataframe_using_project


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



class SequentialLoader:
    """
    Dataloader wrapper around multiple dataloaders, that returns data from them in sequence

    From: https://github.com/PyTorchLightning/pytorch-lightning/issues/12650
    """

    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader


class AbstractNeuronImageFeaturesFromProject(Dataset):

    def __init__(self, project_data: ProjectData, transform=None):
        self.project_data = project_data

    def __len__(self):
        return len(self.project_data.num_frames)

class NeuronImageFeaturesDataModuleFromMultipleProjects(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""
    def __init__(self, batch_size=64, all_project_data: List[ProjectData] = None, num_neurons=None, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=AbstractNeuronImageFeaturesFromProject,
                 assume_all_neurons_correct=False, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.batch_size = batch_size
        self.all_project_data = all_project_data
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.assume_all_neurons_correct = assume_all_neurons_correct

    def setup(self, stage: Optional[str] = None):
        # Split each individually
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.alldata = []
        for project_data in self.all_project_data:
            alldata = self.base_dataset_class(project_data, **self.dataset_kwargs)

            train_fraction = int(len(alldata) * self.train_fraction)
            val_fraction = int(len(alldata) * self.val_fraction)
            splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
            trainset, valset, testset = random_split(alldata, splits)

            # assign to use in dataloaders
            self.train_dataset.append(trainset)
            self.val_dataset.append(valset)
            self.test_dataset.append(testset)

            self.alldata.append(alldata)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.train_dataset
        )
        return SequentialLoader(*dataloaders)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.val_dataset
        )
        return SequentialLoader(*dataloaders)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.test_dataset
        )
        return SequentialLoader(*dataloaders)


class NeuronImageFeaturesDataModuleFromProject(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""
    def __init__(self, batch_size=64, project_data: ProjectData = None, num_neurons=None, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=AbstractNeuronImageFeaturesFromProject,
                 assume_all_neurons_correct=False, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.batch_size = batch_size
        self.project_data = project_data
        self.num_neurons = num_neurons
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.assume_all_neurons_correct = assume_all_neurons_correct

    def setup(self, stage: Optional[str] = None):
        # transform and split
        alldata = self.base_dataset_class(self.project_data, **self.dataset_kwargs)

        train_fraction = int(len(alldata) * self.train_fraction)
        val_fraction = int(len(alldata) * self.val_fraction)
        splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
        trainset, valset, testset = random_split(alldata, splits)

        # assign to use in dataloaders
        self.train_dataset = trainset
        self.val_dataset = valset
        self.test_dataset = testset

        self.alldata = alldata

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def load_data_with_ground_truth():
    ## Load the 4 datasets that have manual annotations
    folder_name = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/"
    fname = os.path.join(folder_name, "round1_worm1/project_config.yaml")
    project_data1 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round1_worm4/project_config.yaml")
    project_data2 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm6/project_config.yaml")
    project_data3 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm3/project_config.yaml")
    project_data4 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    ## Confirm that the tracks are correct
    df1 = correct_tracks_dataframe_using_project(project_data1, overwrite=False, actually_save=False)
    project_data1.final_tracks = df1
    df2 = correct_tracks_dataframe_using_project(project_data2, overwrite=False, actually_save=False)
    project_data2.final_tracks = df2
    df3 = correct_tracks_dataframe_using_project(project_data3, overwrite=False, actually_save=False)
    project_data3.final_tracks = df3
    df4 = correct_tracks_dataframe_using_project(project_data4, overwrite=False, actually_save=False)
    project_data4.final_tracks = df4
    ## Align with the manual annotation .csv file
    project_data1.finished_neurons_column_name = 'Finished?'  # round1 worm 1
    project_data2.finished_neurons_column_name = 'Finished?'  # round1 worm 4
    project_data3.finished_neurons_column_name = 'first 100 frames'  # round2 worm 6
    project_data4.finished_neurons_column_name = 'Finished?'  # round2 worm 3
    project_data1._custom_frame_indices = list(
        range(1000, 3000))  # round1 worm 1; do not include the non-moving portion
    project_data3.num_frames = 100  # round2 worm 6

    all_project_data = [project_data1, project_data2, project_data3, project_data4]

    return all_project_data



if __name__ == "__main__":
    main()

