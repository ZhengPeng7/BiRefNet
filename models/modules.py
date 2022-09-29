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
        if config.dec_non_local:
            self.non_local = NonLocal(channel_in=channel_inter)
        self.conv_out = nn.Conv2d(channel_inter, channel_out, 3, 1, padding=dilation, dilation=dilation)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(channel_inter)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_non_local:
            x = self.non_local(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class DWBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64, dilation=config.dilation):
        super(DWBlk, self).__init__()
        # channel_inter = channel_in // 4
        groups = np.gcd.reduce([channel_in, channel_inter, channel_out])
        self.conv_in = nn.Conv2d(channel_in, channel_inter, 3, 1, padding=dilation, dilation=dilation, groups=groups)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(channel_inter, channel_out, 3, 1, padding=dilation, dilation=dilation, groups=groups)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(channel_inter)
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


class NonLocal(nn.Module):
    def __init__(self, channel_in=512):

        super(NonLocal, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)


    def forward(self, x):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x.size()

        x_query = self.query_transform(x).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        #x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = F.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x = x * x_w
        x = self.conv6(x)

        return x

