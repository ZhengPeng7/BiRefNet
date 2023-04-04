import torch.nn as nn
from models.modules.mlp import MLPLayer


class BlockA(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64, mlp_ratio=4.):
        super(BlockA, self).__init__()
        channel_inter = channel_in
        self.conv = nn.Conv2d(channel_in, channel_inter, 3, 1, 1)
        self.norm1 = nn.LayerNorm(channel_inter)
        self.ffn = MLPLayer(in_features=channel_inter,
                            hidden_features=int(channel_inter * mlp_ratio),
                            act_layer=nn.GELU,
                            drop=0.)
        self.norm2 = nn.LayerNorm(channel_inter)

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