# Load a project and data, then train a Siamese network
import logging
import os

import numpy as np
import torch.optim as optim
from DLC_for_WBFM.utils.nn_utils.data_loading import build_train_loader_batch
from tqdm.auto import tqdm
import torch.nn as nn
import torch

import pandas as pd
from DLC_for_WBFM.utils.nn_utils.siamese import Siamese
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


logging.info("Loading initial data...")
fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-newseg-2021_11_17/project_config.yaml"

project_data = ProjectData.load_final_project_data_from_config(fname)
fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-newseg-2021_11_17/2-training_data/raw/clust_df_dat.pickle"
df = pd.read_pickle(fname)

logging.info("Preprocessing tracklets...")
# Only allow a subset of tracklets to be used
len_thresh = 300
to_keep = df['all_ind_local'].apply(len) > len_thresh
df_long_tracklets = df.loc[to_keep].copy()


logging.info("Initializing network and hyperparameters...")
model = Siamese(embedding_dim=16)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=1.0)
# criterion = torch.jit.script(TripletLoss())
# criterion = torch.jit.script(nn.CrossEntropyLoss())

training_opt = {'df': df_long_tracklets, 'project_data': project_data, 'max_iters': 10, 'batch_size': 8}
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Found cuda!")
clip_value = 5

logging.info("Running network...")
model.train()
all_losses = []
for epoch in tqdm(range(epochs), desc="Epochs"):
    running_loss = []
    train_loader = build_train_loader_batch(**training_opt)

    for step, (anchor_img, positive_img, pos_label, negative_img, neg_label) in enumerate(
            tqdm(train_loader, desc="Training", leave=False)):
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)
        loss = criterion(anchor_out, positive_out, negative_out)

        loss.backward()

        # From: https://stackoverflow.com/questions/66648432/pytorch-test-loss-becoming-nan-after-some-iteration
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        this_loss = loss.cpu().detach().numpy()
        running_loss.append(this_loss)
        if np.isnan(this_loss):
            print("Loss is nan, stopping")
            break
    all_losses.extend(running_loss)

    logging.info("Saving network...")
    fname = os.path.join(project_data.project_dir, f'siamese_epoch{epoch}')
    torch.save(model.state_dict(), fname)
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))
