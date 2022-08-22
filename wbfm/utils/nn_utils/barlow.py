import numpy as np
import torch
from torch import nn, optim
import torchio as tio
import torchvision.transforms as transforms
from pytorch_lightning.core.datamodule import LightningDataModule
from tqdm.auto import tqdm
from wbfm.utils.nn_utils.data_loading import get_bbox_data_for_volume
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from wbfm.utils.nn_utils.utils_fdnc.siamese import Siamese
import math


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = Siamese(embedding_dim=2048)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def embed(self, y1):
        return self.projector(self.backbone(y1))

    def forward(self, y1, y2):
        # Shape of z: neurons x features
        # Because neurons=batch for me, I need to switch the below evaluation
        z1 = self.embed(y1)
        z2 = self.embed(y2)

        # empirical cross-correlation matrix
        c = self.bn(z1) @ self.bn(z2).T
        # c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        # c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class Transform:
    def __init__(self):
        self.transform = tio.transforms.Compose([
            tio.RandomFlip(axes=(1, 2), p=0.5),  # Do not flip z
            tio.RandomBlur(p=1.0),
            # tio.RandomMotion(translation=1, p=1.0),
            tio.RandomElasticDeformation(max_displacement=(1, 5, 5), p=0.5),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            tio.RandomFlip(axes=(1, 2), p=0.5),  # Do not flip z
            tio.RandomBlur(p=0.1),
            # tio.RandomMotion(translation=1, p=0.1),
            tio.RandomElasticDeformation(max_displacement=(1, 5, 5), p=0.1),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)

        sz = y1.shape[0]
        n = nn.InstanceNorm3d(sz, affine=False)
        y1 = n(y1)
        y2 = n(y2)
        return y1, y2


class NeuronImageDataset(Dataset):
    def __init__(self, list_of_neurons_of_volumes):
        self.all_volume_crops = [torch.from_numpy(this_vol.astype(float)) for this_vol in list_of_neurons_of_volumes]
        self.augmentor = Transform()

    def __getitem__(self, idx):
        crops = torch.unsqueeze(self.all_volume_crops[idx], 0)
        # Assume batch=1
        y1, y2 = self.augmentor(torch.squeeze(crops))

        # Normalize; different batch each time
        # Transpose to set the neurons in a volume equal to the batch size (changes per volume)
        # sz = y1.shape[0]
        # n = nn.InstanceNorm3d(sz, affine=False)

        return y1, y2

    def __len__(self):
        return len(self.all_volume_crops)


class NeuronCropImageDataModule(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""

    def __init__(self, batch_size=8, project_data=None, num_frames=100,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=NeuronImageDataset,
                 crop_kwargs=None):
        super().__init__()
        if crop_kwargs is None:
            crop_kwargs = {}
        self.batch_size = batch_size
        self.project_data = project_data
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.crop_kwargs = crop_kwargs

    def setup(self, stage: Optional[str] = None):
        # Get data, then build torch classes
        list_of_neurons_of_volumes = []
        for t in tqdm(range(self.num_frames)):
            vol_dat, _ = get_bbox_data_for_volume(self.project_data, t, **self.crop_kwargs)
            vol_dat = np.stack(vol_dat, 0)
            list_of_neurons_of_volumes.append(vol_dat)

        alldata = NeuronImageDataset(list_of_neurons_of_volumes)

        self.list_of_neurons_of_volumes = list_of_neurons_of_volumes

        # transform and split

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


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases
