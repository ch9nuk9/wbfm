# Load a project and data, then train a Siamese network
import json
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import sacred
import torch
from matplotlib import pyplot as plt

from wbfm.utils.barlow_project.utils.barlow import NeuronCropImageDataModule, BarlowTwins3d
from wbfm.utils.barlow_project.utils.barlow_visualize import visualize_model_performance
from wbfm.utils.barlow_project.utils.siamese import ResidualEncoder3D
from wbfm.utils.projects.finished_project_data import ProjectData
import wandb
from sacred import Experiment

from wbfm.utils.projects.utils_filenames import get_sequential_filename

# !wandb login

ex = Experiment(save_git_info=False)
ex.add_config(# For network
              projector='2048-2048-256', lambd=0.0051, batch_size=1, weight_decay=1e-6,
              epochs=100, print_freq=100, checkpoint_dir='./checkpoint_barlow_small_projector', rank=0,
              lr=0.0001, learning_rate_weights=0.2, learning_rate_biases=0.0048,
              lambd_obj=2.0, train_both_correlations=True, embedding_dim=2048,
              # For data
              target_sz=(4, 128, 128), num_frames=200, train_fraction=0.8, val_fraction=0.1,
              DEBUG=False)


@ex.config
def cfg(**kwargs):
    # Prep model parameters
    args = SimpleNamespace(**kwargs)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    args = _config['args']

    # Load ground truth
    fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/original_training_data/bright_worm5/project_config.yaml"
    project_data1 = ProjectData.load_final_project_data_from_config(fname)

    # Prep data loader
    target_sz = np.array(args.target_sz)
    data_module = NeuronCropImageDataModule(project_data=project_data1, num_frames=args.num_frames, batch_size=1,
                                            train_fraction=args.train_fraction,
                                            val_fraction=args.val_fraction,
                                            crop_kwargs=dict(target_sz=target_sz))
    data_module.setup()
    loader = data_module.train_dataloader()

    # Initialize network
    torch.manual_seed(43)

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # gpu = torch.device("cpu")
    backbone_kwargs = dict(in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz)
    model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Actually train    # import wandb
    # from pytorch_lightning.loggers import WandbLogger

    start_time = time.time()
    stats_file = get_sequential_filename(os.path.join(args.checkpoint_dir, 'stats.json'))
    checkpoint_file = get_sequential_filename(os.path.join(args.checkpoint_dir, 'checkpoint.pth'))

    # with wandb.init(project="barlow_twins", entity="charlesfieseler") as run:
    #     wandb_logger = WandbLogger()
    #     wandb_logger.watch(model, log='all', log_freq=1)

    for epoch in range(0, args.epochs):
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            # Needs to be outside the data loader because the batch dimension isn't added yet
            y1, y2 = torch.transpose(y1, 0, 1).type('torch.FloatTensor'), torch.transpose(y2, 0, 1).type(
                'torch.FloatTensor')
            y1 = y1.to(gpu)
            y2 = y2.to(gpu)

            # adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            loss = model.forward(y1, y2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    with open(stats_file, 'w') as f:
                        print(json.dumps(stats), file=f)

                    # Also plot embedding
                    with torch.no_grad():
                        c = model.calculate_correlation_matrix(y1, y2)
                        visualize_model_performance(c)
                        fig_fname = os.path.join(args.checkpoint_dir, f'correlation_matrix_{step}.png')
                        plt.savefig(fig_fname)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, checkpoint_file)

    # Final saving
    if args.rank == 0:
        # save final model
        fname = get_sequential_filename(args.checkpoint_dir + '/resnet50.pth')
        torch.save(model.state_dict(), fname)

    # Also save the args namespace
    fname = get_sequential_filename(args.checkpoint_dir + '/args.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(args, f)
