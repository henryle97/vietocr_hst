import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
from vietocr.model.backbone.vgg_2015 import VGG_FeatureExtractor

class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif backbone == 'vgg_crnn':
            self.model = VGG_FeatureExtractor(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
