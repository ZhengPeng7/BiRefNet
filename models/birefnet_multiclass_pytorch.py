import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class BiRefNetMultiClass(nn.Module):
    def __init__(self, num_classes=21, backbone_pretrained=True):
        super().__init__()
        # Backbone: ResNet-50
        backbone = resnet50(pretrained=backbone_pretrained)
        
        # Extract multi-scale features
        self.conv1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Feature Pyramid Decoder
        self.decoder4 = self._decoder_block(2048, 1024)
        self.decoder3 = self._decoder_block(1024, 512)
        self.decoder2 = self._decoder_block(512, 256)
        self.decoder1 = self._decoder_block(256, 64)

        # Final classification layer
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x, labels=None):
        # Encoder (Backbone)
        x0 = self.conv1(x)   # [B, 64, H/4, W/4]
        x1 = self.layer1(x0) # [B, 256, H/4, W/4]
        x2 = self.layer2(x1) # [B, 512, H/8, W/8]
        x3 = self.layer3(x2) # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3) # [B, 2048, H/32, W/32]

        # Decoder with skip connections
        d4 = self.decoder4(x4) + x3    # [B, 1024, H/16, W/16]
        d3 = self.decoder3(d4) + x2    # [B, 512, H/8, W/8]
        d2 = self.decoder2(d3) + x1    # [B, 256, H/4, W/4]
        d1 = self.decoder1(d2)         # [B, 64, H/2, W/2]

        # Final upsampling to input size
        logits = self.classifier(d1)   # [B, num_classes, H/2, W/2]
        logits = F.interpolate(logits, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Compute loss if labels provided
        outputs = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs["loss"] = loss

        return outputs
