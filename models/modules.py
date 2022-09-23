import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from functools import partial
from einops import rearrange

from config import Config


config = Config()


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, dilation=config.dilation):
        super(ResBlk, self).__init__()
        if dilation == 1:
            self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        elif dilation == 2:
            self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, dilation, dilation=dilation)
        self.relu_in = nn.ReLU(inplace=True)
        if dilation == 1:
            self.conv_in = nn.Conv2d(64, channel_out, 3, 1, 1)
        elif dilation == 2:
            self.conv_in = nn.Conv2d(64, channel_out, 3, 1, dilation, dilation=dilation)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x
