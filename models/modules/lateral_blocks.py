import torch.nn as nn

from config import Config


config = Config()


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, ks=1, s=1, p=0):
        super(BasicLatBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ks, s, p)

    def forward(self, x):
        x = self.conv(x)
        return x
