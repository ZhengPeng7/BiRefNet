import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.deform_conv import DeformableConv2d


class _ASPPModule(nn.Module):
    def __init__(self, channel_in, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(channel_in, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, channel_in=64, output_stride=16):
        super(ASPP, self).__init__()
        self.down_scale = 1
        self.channel_inter = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(channel_in, self.channel_inter, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_in, self.channel_inter, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.channel_inter),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.channel_inter * 5, channel_in, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


##################### Deformable
class _ASPPModuleDeformable(nn.Module):
    def __init__(self, channel_in, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(channel_in, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, channel_in=64):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        self.channel_inter = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(channel_in, self.channel_inter, 1, padding=0)
        self.aspp2 = _ASPPModuleDeformable(channel_in, self.channel_inter, 3, padding=1)
        self.aspp3 = _ASPPModuleDeformable(channel_in, self.channel_inter, 3, padding=1)
        self.aspp4 = _ASPPModuleDeformable(channel_in, self.channel_inter, 3, padding=1)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_in, self.channel_inter, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.channel_inter),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.channel_inter * 5, channel_in, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

