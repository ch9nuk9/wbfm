from dataclasses import dataclass

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F


class NeuronEmbeddingModel(LightningModule):

    def __init__(self, feature_dim=840, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)
        self.fc4 = nn.Linear(120, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y = labels.to(self.device)
        X = inputs.to(self.device)

        # zero the parameter gradients
        # self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, prog_bar=True)

        # loss.backward()
        # self.optimizer.step()
        # if batch_idx % 1000 == 999:
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def embed(self, x):
        # Everything but the last layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
