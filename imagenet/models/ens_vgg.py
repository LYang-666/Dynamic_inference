'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d,  NasBatchNorm2d,  noise_Conv2d, noise_Conv2d1, noise_Linear
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear, aw_BatchNorm2d, aw_DownsampleA

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, cfg):
        super(VGG, self).__init__()

        self.ch_width = [64, 128, 256, 256,  512, 512, 512, 512, 4096, 4096]
        self.idx = 0
        self.features = self.make_layers(cfg, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nas_noise_Linear(self.ch_width[7]*7*7, self.ch_width[8]),
            nn.ReLU(True),
            nn.Dropout(),
            nas_noise_Linear(self.ch_width[8], self.ch_width[9]),
            nn.ReLU(True),
            nas_noise_Linear(self.ch_width[9], 1000),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        i = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i == 0:
                    conv2d = nas_noise_Conv2d(in_channels, self.ch_width[0], kernel_size=3, padding=1)
                else:
                    conv2d = nas_noise_Conv2d(self.ch_width[i-1], self.ch_width[i], kernel_size=3, padding=1)

                if batch_norm:
                    layers += [conv2d, NasBatchNorm2d(self.ch_width[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                i+=1
                in_channels = v
        return nn.Sequential(*layers)

    def update_model(self):

        l = 0
        for name, m in self.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if l > 0:
                    m.in_channels = self.ch_width[l-1]
  
                m.out_channels = self.ch_width[l]
                l += 1
       
            if isinstance(m, nn.Linear):

                if l == 10:
                    m.in_features = self.ch_width[-1]
          
                else:
                    m.in_features = self.ch_width[l-1]
                    m.out_features = self.ch_width[l]
                    if l == 8:
                        m.in_features = self.ch_width[l-1]*7*7
                l+=1
            if isinstance(m, NasBatchNorm2d):

                m.num_features = self.ch_width[l-1]
                m.update_idx(self.idx)
cfg = {         
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def ens_vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg['A'])


def ens_vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg['A'])


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))