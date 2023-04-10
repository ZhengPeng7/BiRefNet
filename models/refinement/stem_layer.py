import torch.nn as nn
from models.modules.utils import build_act_layer, build_norm_layer


class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_channels=3+1,
                 inter_channels=48,
                 out_channels=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               inter_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm1 = build_norm_layer(
            inter_channels, norm_layer, 'channels_first', 'channels_first'
        )
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(inter_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = build_norm_layer(
            out_channels, norm_layer, 'channels_first', 'channels_first'
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
