import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50

from models.modules import ResBlk
from models.bb_pvtv2 import pvt_v2_b2
from config import Config


class BSL(nn.Module):
    def __init__(self):
        super(BSL, self).__init__()
        self.config = Config()
        self.epoch = 1
        bb = self.config.bb
        if bb == 'cnn-vgg16':
            bb_net = list(vgg16(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_net[:4],
                'conv2': bb_net[4:9],
                'conv3': bb_net[9:16],
                'conv4': bb_net[16:23]
            })
        elif bb == 'cnn-vgg16bn':
            bb_net = list(vgg16_bn(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_net[:6],
                'conv2': bb_net[6:13],
                'conv3': bb_net[13:23],
                'conv4': bb_net[23:33]
            })
        elif bb == 'cnn-resnet50':
            bb_net = list(resnet50(pretrained=True).children())
            bb_convs = OrderedDict({
                'conv1': nn.Sequential(*bb_net[0:3]),
                'conv2': bb_net[4],
                'conv3': bb_net[5],
                'conv4': bb_net[6]
            })
        elif bb == 'trans-pvt':
            self.bb = pvt_v2_b2()
            if self.config.pvt_weights:
                save_model = torch.load(self.config.pvt_weights)
                model_dict = self.bb.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.bb.load_state_dict(model_dict)

        if 'cnn-' in bb:
            self.bb = nn.Sequential(bb_convs)
        lateral_channels_in = {
            'cnn-vgg16': [512, 256, 128, 64],
            'cnn-vgg16bn': [512, 256, 128, 64],
            'cnn-resnet50': [1024, 512, 256, 64],
            'trans-pvt': [512, 320, 128, 64],
        }

        if self.config.dec_blk == 'ResBlk':
            DecBlk = ResBlk

        self.top_layer = DecBlk(lateral_channels_in[bb][0], lateral_channels_in[bb][1])

        self.dec_layer4 = DecBlk(lateral_channels_in[bb][1], lateral_channels_in[bb][1])
        self.lat_layer4 = nn.Conv2d(lateral_channels_in[bb][1], lateral_channels_in[bb][1], 1, 1, 0)

        self.dec_layer3 = DecBlk(lateral_channels_in[bb][1], lateral_channels_in[bb][2])
        self.lat_layer3 = nn.Conv2d(lateral_channels_in[bb][2], lateral_channels_in[bb][2], 1, 1, 0)

        self.dec_layer2 = DecBlk(lateral_channels_in[bb][2], lateral_channels_in[bb][3])
        self.lat_layer2 = nn.Conv2d(lateral_channels_in[bb][3], lateral_channels_in[bb][3], 1, 1, 0)

        self.dec_layer1 = DecBlk(lateral_channels_in[bb][3], lateral_channels_in[bb][3]//2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(lateral_channels_in[bb][3]//2, 1, 1, 1, 0))

        if self.config.freeze_bb:
            print(self.named_parameters())
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        ########## Encoder ##########

        if 'trans' in self.config.bb:
            x1, x2, x3, x4 = self.bb(x)
        else:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)

        p4 = self.top_layer(x4)

        ########## Decoder ##########
        scaled_preds = []

        p4 = self.dec_layer4(p4)
        p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        p3 = p4 + self.lat_layer4(x3)

        p3 = self.dec_layer3(p3)
        p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        p2 = p3 + self.lat_layer3(x2)

        p2 = self.dec_layer2(p2)
        p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2 + self.lat_layer2(x1)

        p1 = self.dec_layer1(p1)
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        p1_out = self.conv_out1(p1)
        scaled_preds.append(p1_out)

        return scaled_preds
