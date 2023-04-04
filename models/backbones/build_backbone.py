import torch
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights, resnet50, ResNet50_Weights
from models.backbones.pvt_v2 import pvt_v2_b2, pvt_v2_b5
from models.backbones.swin_v1 import SwinB, SwinL
from config import Config


config = Config()

def build_backbone(bb_name, pretrained=True):
    if bb_name == 'vgg16':
        bb_net = list(vgg16(pretrained=VGG16_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:4], 'conv2': bb_net[4:9], 'conv3': bb_net[9:16], 'conv4': bb_net[16:23]}))
    elif bb_name == 'vgg16bn':
        bb_net = list(vgg16_bn(pretrained=VGG16_BN_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:6], 'conv2': bb_net[6:13], 'conv3': bb_net[13:23], 'conv4': bb_net[23:33]}))
    elif bb_name == 'resnet50':
        bb_net = list(resnet50(pretrained=ResNet50_Weights.DEFAULT if pretrained else None).children())
        bb = nn.Sequential(OrderedDict({'conv1': nn.Sequential(*bb_net[0:3]), 'conv2': bb_net[4], 'conv3': bb_net[5], 'conv4': bb_net[6]}))
    else:
        bb = load_weights(eval('{}()'.format(bb_name)), bb_name) if pretrained else eval('{}()'.format(bb_name))
    return bb

def load_weights(model, model_name):
    save_model = torch.load(config.weights[model_name])
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
