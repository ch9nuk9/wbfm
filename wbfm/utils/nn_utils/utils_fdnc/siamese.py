import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self, embedding_dim=4096):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm3d(1),
            nn.Conv3d(1, 64, 4, padding=3),  # 64@8*64*64
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 64@4*32*32
            nn.Conv3d(64, 128, 4),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 128@2*16*16
            nn.Conv3d(128, 128, (1, 2, 2)),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(),    # 128@1*14*14
        )
        self.projection = nn.Sequential(nn.Linear(128 * 1 * 14 * 14, embedding_dim), nn.Sigmoid())
        self.out = nn.Linear(embedding_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x
