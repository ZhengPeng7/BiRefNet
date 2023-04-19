import torch
import torch.nn as nn
from models.modules.aspp import ASPP, ASPPDeformable
from models.modules.attentions import PSA, SGE
from config import Config


config = Config()


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicDecBlk, self).__init__()
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == 'ASPPDeformable':
            self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class ResBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, inter_channels=64):
        super(ResBlk, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64

        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu_in = nn.ReLU(inplace=True)

        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == 'ASPPDeformable':
            self.dec_att = ASPPDeformable(in_channels=inter_channels)

        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        _x = self.conv_resi(x)
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x + _x


class HierarAttDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, inter_channels=64):
        super(HierarAttDecBlk, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.split_y = 8     # must be divided by channels of all intermediate features
        self.split_x = 8

        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)

        self.psa = PSA(inter_channels*self.split_y*self.split_x, S=config.batch_size)
        self.sge = SGE(groups=config.batch_size)

        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(in_channels=inter_channels)
        elif config.dec_att == 'ASPPDeformable':
            self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        N, C, H, W = x.shape
        x_patchs = x.reshape(N, -1, H//self.split_y, W//self.split_x)

        # Hierarchical attention: group attention X patch spatial attention
        x_patchs = self.psa(x_patchs)   # Group Channel Attention -- each group is a single image
        x_patchs = self.sge(x_patchs)   # Patch Spatial Attention
        x = x.reshape(N, C, H, W)
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        x = self.conv_out(x)
        return x
