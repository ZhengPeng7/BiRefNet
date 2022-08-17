import os


class Config():
    def __init__(self) -> None:
        # Backbone
        self.bb = ['cnn-vgg16', 'cnn-vgg16bn', 'cnn-resnet50', 'trans-pvt'][3]
        self.pvt_weights = ['../bb_weights/pvt_v2_b2.pth', ''][0]
        # BN
        self.use_bn = self.bb not in ['cnn-vgg16']
        # Augmentation
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]

        # Components

        # Training

        # Loss

        # others
