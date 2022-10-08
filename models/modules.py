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
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64, dilation=config.dilation):
        super(ResBlk, self).__init__()
        channel_inter = channel_in // 4 if config.dec_channel_inter == 'adap' else 64
        self.conv_in = nn.Conv2d(channel_in, channel_inter, 3, 1, padding=dilation, dilation=dilation)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att:
            self.dec_att = AttentionModule(channel_in=channel_inter)
        self.conv_out = nn.Conv2d(channel_inter, channel_out, 3, 1, padding=dilation, dilation=dilation)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(channel_inter)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channel_in=64):
        super(AttentionModule, self).__init__()
        self.conv0 = nn.Conv2d(channel_in, channel_in, 5, padding=2, groups=channel_in)
        self.conv0_1 = nn.Conv2d(channel_in, channel_in, (1, 7), padding=(0, 3), groups=channel_in)
        self.conv0_2 = nn.Conv2d(channel_in, channel_in, (7, 1), padding=(3, 0), groups=channel_in)

        self.conv1_1 = nn.Conv2d(channel_in, channel_in, (1, 11), padding=(0, 5), groups=channel_in)
        self.conv1_2 = nn.Conv2d(channel_in, channel_in, (11, 1), padding=(5, 0), groups=channel_in)

        self.conv2_1 = nn.Conv2d(
            channel_in, channel_in, (1, 21), padding=(0, 10), groups=channel_in)
        self.conv2_2 = nn.Conv2d(
            channel_in, channel_in, (21, 1), padding=(10, 0), groups=channel_in)
        self.conv3 = nn.Conv2d(channel_in, channel_in, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

