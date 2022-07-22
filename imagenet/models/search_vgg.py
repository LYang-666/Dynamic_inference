'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d,  NasBatchNorm2d,  noise_Conv2d, noise_Conv2d1, noise_Linear
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear1, aw_BatchNorm2d, aw_DownsampleA
import torch.utils.model_zoo as model_zoo
import numpy as np
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.ch_width = [64, 128, 256, 256,  512, 512, 512, 512, 4096, 4096]
        self.ch_index = [[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]]

        # self.ch_width = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096]
        # self.ch_index = [[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]]
        self.features = self.make_layers(cfg, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            aw_noise_Linear1(self.ch_width[7]* 7 * 7, self.ch_width[8], channel_index_in=self.ch_index[7],
                                channel_index_out=self.ch_index[8]),
            nn.ReLU(True),
            nn.Dropout(),
            aw_noise_Linear1(self.ch_width[8], self.ch_width[9], channel_index_in=self.ch_index[8],
                                channel_index_out=self.ch_index[9]),
            nn.ReLU(True),
            nn.Dropout(),
            aw_noise_Linear1(self.ch_width[9], 1000, channel_index_in=self.ch_index[9]),
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
                    conv2d = aw_noise_Conv2d(in_channels, self.ch_width[0] , kernel_size=3, padding=1,channel_index_in=self.ch_index[0])
                else:
                    conv2d = aw_noise_Conv2d(self.ch_width[i-1], self.ch_width[i] , kernel_size=3, padding=1,channel_index_in=self.ch_index[i-1],
                                            channel_index_out=self.ch_index[i])

                if batch_norm:
                    layers += [conv2d, aw_BatchNorm2d(self.ch_width[i], channel_index=self.ch_index[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                i+=1
                in_channels = v
        return nn.Sequential(*layers)

    def update_model(self, layer):

        l = 0
        for name, m in self.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if l > 0 and l==(layer+1):
                    m.in_channels = self.ch_width[l-1]
                    m.channel_index_in = self.ch_index[l-1]
                    m._update_n_channels(self.ch_index[l-1], in_ch=True)
                if l == layer:
                    m.out_channels = self.ch_width[l]
                    m.channel_index_out = self.ch_index[l]
                    m._update_n_channels(self.ch_index[l], out_ch=True)                    

                l += 1
       
            if isinstance(m, nn.Linear):
                if l == 10:
                    m.in_features = self.ch_width[-1]
                    m.channel_index = self.ch_index[-1]
                    m._update_n_channels(self.ch_index[-1], in_ch=True)   
                else:
                    if l == 8:
                        b = [-1]
                        m.in_features = self.ch_width[l-1] *7 * 7
                        
                        for i in self.ch_index[l-1]:
                            if i is not -1:
                                for j in np.arange(i*49, i*49+49):
                                    b.append(j)
                        m.channel_index = b
                        m._update_n_channels(b, in_ch=True)     
                        m.out_features = self.ch_width[l]
                        m.channel_index_out = self.ch_index[l]
                        m._update_n_channels(self.ch_index[l], out_ch=True)       
                    else:           
                        m.in_features = self.ch_width[l-1]
                        m.channel_index = self.ch_index[l-1]
                        m._update_n_channels(self.ch_index[l-1], in_ch=True)
                        m.out_features = self.ch_width[l]
                        m.channel_index_out = self.ch_index[l]
                        m._update_n_channels(self.ch_index[l], out_ch=True)
                l+=1
            if isinstance(m, nn.BatchNorm2d):

                if l == (layer+1):
                    m.num_features = self.ch_width[l-1]
                    m.channel_index = self.ch_index[l-1]
                    m._update_n_channels(self.ch_index[l-1])  
cfgs = {         
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(cfgs[cfg])
    if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls[arch]))


       #load part of the weights from model zoo
       pretrained_dict = model_zoo.load_url(model_urls[arch])
       model_dict = model.state_dict()
       # 1. filter out unnecessary keys
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       # 2. overwrite entries in the existing state dict
       model_dict.update(pretrained_dict) 
       # 3. load the new state dict
       model.load_state_dict(model_dict)
                          
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def search_vgg11_bn(pretrained=True, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def search_vgg16(pretrained=True, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
