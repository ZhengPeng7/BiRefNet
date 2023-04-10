import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50

from config import Config
from dataset import class_labels_TR_sorted
from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.modules.ing import *
from models.refinement.stem_layer import StemLayer


class RefinerPVTInChannels4(nn.Module):
    def __init__(self, in_channels=3+1):
        super(RefinerPVTInChannels4, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, params_settings='in_channels=4')

        lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
        }
        channels = lateral_channels_in_collection[self.config.bb]
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        self.decoder = Decoder(channels)

        if 0:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        ########## Encoder ##########
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)

        x4 = self.squeeze_module(x4)

        ########## Decoder ##########

        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Refiner(nn.Module):
    def __init__(self, in_channels=3+1):
        super(Refiner, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.stem_layer = StemLayer(in_channels=in_channels, inter_channels=48, out_channels=3)
        self.bb = build_backbone(self.config.bb)

        lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
        }
        channels = lateral_channels_in_collection[self.config.bb]
        self.squeeze_module = BasicDecBlk(channels[0], channels[0])

        self.decoder = Decoder(channels)

        if 0:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        x = self.stem_layer(x)
        ########## Encoder ##########
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)

        x4 = self.squeeze_module(x4)

        ########## Decoder ##########

        features = [x, x1, x2, x3, x4]
        scaled_preds = self.decoder(features)

        return scaled_preds


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        DecoderBlock = eval('BasicDecBlk')
        LateralBlock = eval('BasicLatBlk')

        self.decoder_block4 = DecoderBlock(channels[0], channels[1])
        self.decoder_block3 = DecoderBlock(channels[1], channels[2])
        self.decoder_block2 = DecoderBlock(channels[2], channels[3])
        self.decoder_block1 = DecoderBlock(channels[3], channels[3]//2)

        self.lateral_block4 = LateralBlock(channels[1], channels[1])
        self.lateral_block3 = LateralBlock(channels[2], channels[2])
        self.lateral_block2 = LateralBlock(channels[3], channels[3])

        if self.config.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3]//2, 1, 1, 1, 0))

    def forward(self, features):
        x, x1, x2, x3, x4 = features
        outs = []
        p4 = self.decoder_block4(x4)
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        p3 = self.decoder_block3(_p3)
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        p2 = self.decoder_block2(_p2)
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        p1_out = self.conv_out1(_p1)

        if self.config.ms_supervision:
            outs.append(self.conv_ms_spvn_4(p4))
            outs.append(self.conv_ms_spvn_3(p3))
            outs.append(self.conv_ms_spvn_2(p2))
        outs.append(p1_out)
        return outs


class RefUNet(nn.Module):
    # Refinement
    def __init__(self, in_channels=3+1):
        super(RefUNet, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #####
        self.decoder_5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #####
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_d0 = nn.Conv2d(64, 1, 3, 1, 1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        outs = []
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        hx = x

        hx1 = self.encoder_1(hx)
        hx2 = self.encoder_2(hx1)
        hx3 = self.encoder_3(hx2)
        hx4 = self.encoder_4(hx3)

        hx = self.decoder_5(self.pool4(hx4))
        hx = torch.cat((self.upscore2(hx), hx4), 1)

        d4 = self.decoder_4(hx)
        hx = torch.cat((self.upscore2(d4), hx3), 1)

        d3 = self.decoder_3(hx)
        hx = torch.cat((self.upscore2(d3), hx2), 1)

        d2 = self.decoder_2(hx)
        hx = torch.cat((self.upscore2(d2), hx1), 1)

        d1 = self.decoder_1(hx)

        x = self.conv_d0(d1)
        outs.append(x)
        return outs
