import torch.nn as nn
from models.modules.mlp import MLPLayer


class BlockA(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64, mlp_ratio=4.):
        super(BlockA, self).__init__()
        inter_channels = in_channels
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.norm1 = nn.LayerNorm(inter_channels)
        self.ffn = MLPLayer(in_features=inter_channels,
                            hidden_features=int(inter_channels * mlp_ratio),
                            act_layer=nn.GELU,
                            drop=0.)
        self.norm2 = nn.LayerNorm(inter_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        _x = self.conv(x)
        _x = _x.flatten(2).transpose(1, 2)
        _x = self.norm1(_x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = x + _x
        _x1 = self.ffn(x)
        _x1 = self.norm2(_x1)
        _x1 = _x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x + _x1
        return x