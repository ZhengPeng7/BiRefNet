import torch.nn as nn
from models.modules.aspp import ASPP, ASPPDeformable
from config import Config


config = Config()


class BasicDecBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64):
        super(BasicDecBlk, self).__init__()
        channel_inter = channel_in // 4 if config.dec_channel_inter == 'adap' else 64
        self.conv_in = nn.Conv2d(channel_in, channel_inter, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(channel_in=channel_inter)
        elif config.dec_att == 'ASPPDeformable':
            self.dec_att = ASPPDeformable(channel_in=channel_inter)
        self.conv_out = nn.Conv2d(channel_inter, channel_out, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(channel_inter)
        self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x
