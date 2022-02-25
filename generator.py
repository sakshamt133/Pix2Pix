import torch.nn as nn
import torch
from block import UpSample, DownSample


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.down1 = DownSample(in_channels, 8, (3, 3))
        self.down2 = DownSample(8, 16, (5, 5))
        self.down3 = DownSample(16, 32, (3, 3), (2, 2))
        self.down4 = DownSample(32, 64, (3, 3), (2, 2))
        self.down5 = DownSample(64, 128, (3, 3))
        self.up1 = UpSample(128, 64, (3, 3))
        self.up2 = UpSample(64, 32, (3, 3), (2, 2))
        self.up3 = UpSample(32, 16, (4, 4), (2, 2))
        self.up4 = UpSample(16, 8, (5, 5))
        self.up5 = UpSample(8, 3, (3, 3))

    def forward(self, x):
        down1 = self.down1(x)
        out2 = self.down2(down1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        out = self.down5(out4)
        out = self.up1(out)
        out = self.up2(out + out4)
        out = out + out3
        out = self.up3(out)
        out = self.up4(out)
        out = self.up5(out + down1)
        return out

# g = Generator(3)
# temp = torch.randn((1, 3, 26, 26))
# out = g(temp)
# print(out.shape)
