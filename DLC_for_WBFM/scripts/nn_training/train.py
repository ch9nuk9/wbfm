# Load a project and data, then train a Siamese network
import logging
import os
import numpy as np
import torch.optim as optim
from DLC_for_WBFM.utils.nn_utils.losses import ArcMarginProduct
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename
from tqdm.auto import tqdm
import torch.nn as nn
import torch
from DLC_for_WBFM.utils.nn_utils.data_loading import NeuronTripletDataset
from torch.utils.data import DataLoader
import pandas as pd
from DLC_for_WBFM.utils.nn_utils.siamese import Siamese
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
import wandb
# !wandb login


seed = 43

# random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.info("Loading initial data...")
fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-newseg-2021_11_17/project_config.yaml"

project_data = ProjectData.load_final_project_data_from_config(fname)
fname = "/scratch/zimmer/Charles/dlc_stacks/worm3-newseg-2021_11_17/2-training_data/raw/clust_df_dat.pickle"
df = pd.read_pickle(fname)

## NOTE: use a previously saved and preprocessed dataset
# logging.info("Preprocessing tracklets...")
# Only allow a subset of tracklets to be used
# len_thresh = 300
# to_keep = df['all_ind_local'].apply(len) > len_thresh
# df_long_tracklets = df.loc[to_keep].copy()

##
logging.info("Initializing network and hyperparameters...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Found cuda!")
else:
    print("Did not find cuda")

epochs = 100
embedding_dim = 16
num_labels = 1028  # Tracklets are treated as different neurons

model = Siamese(embedding_dim=embedding_dim)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
metric = ArcMarginProduct(embedding_dim, num_labels, easy_margin=True, device=device)
# metric = None
criterion = nn.TripletMarginLoss(margin=1.0)
# criterion = torch.jit.script(TripletLoss())
# criterion = torch.jit.script(nn.CrossEntropyLoss())
training_folder = os.path.join(project_data.project_dir, 'nn_training')
training_opt = {'batch_size': 8}
training_dataset = NeuronTripletDataset(training_folder, remap_labels=True)
train_loader = DataLoader(training_dataset, **training_opt)

# clip_value = 5

##
logging.info("Running network...")
model.train()
all_losses = []


def _cast_images_and_calc_loss(anchor_img, positive_img, negative_img, pos_label, neg_label, criterion):
    anchor_img = anchor_img.to(device)
    positive_img = positive_img.to(device)
    negative_img = negative_img.to(device)
    optimizer.zero_grad()
    anchor_out = model(anchor_img)
    positive_out = model(positive_img)
    negative_out = model(negative_img)
    if metric is not None:
        anchor_out = metric(anchor_out, pos_label)
        positive_out = metric(positive_out, pos_label)
        negative_out = metric(negative_out, neg_label)
    loss = criterion(anchor_out, positive_out, negative_out)
    loss.backward()

    return loss


with wandb.init(project="debuggingnn", entity="charlesfieseler"):
    wandb.watch(model, log='all', log_freq=1)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []

        for step, (anchor_img, positive_img, pos_label, negative_img, neg_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            loss = _cast_images_and_calc_loss(anchor_img, positive_img, negative_img,
                                              pos_label=pos_label,
                                              neg_label=neg_label,
                                              criterion=criterion)

            # From: https://stackoverflow.com/questions/66648432/pytorch-test-loss-becoming-nan-after-some-iteration
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            this_loss = loss.cpu().detach().numpy()
            running_loss.append(this_loss)
            if np.isnan(this_loss):
                print("Loss is nan, stopping")
                break
            wandb.log({'Loss': this_loss})
        all_losses.extend(running_loss)

        logging.info("Saving network...")
        if epoch % 10 == 0:
            fname = os.path.join(project_data.project_dir, f'siamese_epoch{epoch}')
            fname = get_sequential_filename(fname)
            torch.save(model.state_dict(), fname)
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))
