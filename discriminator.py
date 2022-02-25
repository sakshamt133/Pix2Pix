import torch.nn as nn
import torch
from block import DownSample


class PatchGAN(nn.Module):
    def __init__(self, img_channels):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            DownSample(img_channels * 2, 8, (3, 3)),
            DownSample(8, 16, (3, 3)),
            DownSample(16, 32, (5, 5)),
            DownSample(32, 64, (7, 7)),
            nn.Conv2d(64, 32, (5, 5), (2, 2), (1, 1)),
            nn.Conv2d(32, 1, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, real, fake):
        x = torch.cat([real, fake], dim=1)
        print(x.shape)
        return self.model(x)

