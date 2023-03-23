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
from models.dec_pvtv2 import pvt_v2_b2_decoder
from config import Config
from dataset import class_labels_TR_sorted


class PVTVP(nn.Module):
    def __init__(self):
        super(PVTVP, self).__init__()
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

        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(lateral_channels_in[bb][0], len(class_labels_TR_sorted))
            )

        self.decoder = pvt_v2_b2_decoder()

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

        if self.training and self.config.auxiliary_classification:
            class_preds = self.cls_head(self.avgpool(x4).view(x4.shape[0], -1))


        ########## Decoder ##########
        scaled_preds = self.decoder(x4, [x3, x2, x1])[-1:]

        # # refine patch-level segmentation
        # if self.config.refine:
        #     p0 = self.refiner(p1_out)

        if self.config.auxiliary_classification:
            return scaled_preds, class_preds
        else:
            return scaled_preds
