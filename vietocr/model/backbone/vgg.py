import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter


class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(Vgg, self).__init__()

        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=pretrained)

        pool_idx = 0
        
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):        
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
 
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)    # B*C*H*W
        # from IPython import embed; embed()

#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2) # B*C*W*H
        conv = conv.flatten(2)          # B*C* (WxH)
        conv = conv.permute(-1, 0, 1)   # T*B*C


        return conv

def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg11_bn', ss, ks, hidden, pretrained, dropout)

def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', ss, ks, hidden, pretrained, dropout)
   
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    ss = [[2, 2],
        [2, 2],
        [2, 1],
        [2, 1],
        [1, 1]]
    ks = [[2, 2],
          [2, 2],
          [2, 1],
          [2, 1],
          [1, 1]]
    import time


    model = vgg19_bn(ss, ks, hidden=256)
    t1 = time.time()
    print(model(torch.randn((1, 3, 32, 128))).shape)
    print(time.time() - t1)
    print(count_parameters(model))

    # print(model)
    #
    # print("Num parameters: ", count_parameters(model))