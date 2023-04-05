import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from config import Config


config = Config()


class BasicLatBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64):
        super(BasicLatBlk, self).__init__()
        channel_inter = channel_in // 4 if config.dec_channel_inter == 'adap' else 64
        self.conv = nn.Conv2d(channel_in, channel_out, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x
