
from torch.utils.data import DataLoader
from fDNC.src.model import NIT_Registration, neuron_data_pytorch
import torch
import math
import time
import os
import numpy as np
from tqdm.auto import tqdm

DATA_PATH="/scratch/zimmer/fieseler/github_repos/fDNC_Neuron_ID/Data"
train_path = f"{DATA_PATH}/train_synthetic"
eval_path = f"{DATA_PATH}/validation_synthetic"

batch_size = 64
data_mode = 'all'

n_hidden = 48
f_trans = 1
p_rotate = 1
n_layer = 8


# loading the data
train_data = neuron_data_pytorch(train_path, batch_sz=batch_size, shuffle=True, rotate=True, mode=data_mode)
dev_data = neuron_data_pytorch(eval_path, batch_sz=batch_size, shuffle=False, rotate=True, mode=data_mode)


train_data_loader = DataLoader(train_data, shuffle=False, num_workers=1, collate_fn=train_data.custom_collate_fn)
dev_data_loader = DataLoader(dev_data, shuffle=False, num_workers=1, collate_fn=dev_data.custom_collate_fn)
# device = torch.device(f"cuda:{args.cuda_device_index}" if cuda else "cpu")
model = NIT_Registration(input_dim=3, n_hidden=n_hidden, n_layer=n_layer, p_rotate=p_rotate,
                             feat_trans=f_trans, cuda=False)

for t in train_data:
    print(t)
    out = model(t)
    print(out)
    break
