import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.nn.init as init
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d,  NasBatchNorm2d,  noise_Conv2d, noise_Conv2d1, noise_Linear
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear1, aw_BatchNorm2d, aw_DownsampleA
import torch.utils.model_zoo as model_zoo
import numpy as np

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.ch_width = [64, 192, 384, 256, 256, 4096, 4096]
        self.ch_index = [[-1],[-1],[-1],[-1],[-1],[-1],[-1]]

        self.idx = 0
        self.features = nn.Sequential(
            aw_noise_Conv2d(3, self.ch_width[0], kernel_size=11, stride=4, padding=2, channel_index_in=self.ch_index[0]),
            # aw_BatchNorm2d(self.ch_width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            aw_noise_Conv2d(self.ch_width[0], self.ch_width[1], kernel_size=5, padding=2,channel_index_in=self.ch_index[0],
                                            channel_index_out=self.ch_index[1]),
            # aw_BatchNorm2d(self.ch_width[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            aw_noise_Conv2d(self.ch_width[1], self.ch_width[2], kernel_size=3, padding=1,channel_index_in=self.ch_index[1],
                                            channel_index_out=self.ch_index[2]),
            # aw_BatchNorm2d(self.ch_width[2]),
            nn.ReLU(inplace=True),
            aw_noise_Conv2d(self.ch_width[2], self.ch_width[3], kernel_size=3, padding=1,channel_index_in=self.ch_index[2],
                                            channel_index_out=self.ch_index[3]),
            # aw_BatchNorm2d(self.ch_width[3]),
            nn.ReLU(inplace=True),
            aw_noise_Conv2d(self.ch_width[3], self.ch_width[4], kernel_size=3, padding=1,channel_index_in=self.ch_index[3],
                                            channel_index_out=self.ch_index[4]),
            # aw_BatchNorm2d(self.ch_width[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            aw_noise_Linear1(self.ch_width[4] * 6 * 6, self.ch_width[5], channel_index_in=self.ch_index[4],
                                channel_index_out=self.ch_index[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            aw_noise_Linear1(self.ch_width[5], self.ch_width[6], channel_index_in=self.ch_index[5],
                                channel_index_out=self.ch_index[6]),
            nn.ReLU(inplace=True),
            aw_noise_Linear1(self.ch_width[6], num_classes, channel_index_in=self.ch_index[6]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
                if l == 7:
                    m.in_features = self.ch_width[-1]
                    m.channel_index = self.ch_index[-1]
                    m._update_n_channels(self.ch_index[-1], in_ch=True)   
                else:
                    if l == 5:
                        b = [-1]
                        m.in_features = self.ch_width[l-1] *6 * 6
                        
                        for i in self.ch_index[l-1]:
                            if i is not -1:
                                for j in np.arange(i*36, i*36+36):
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


def search_alexnet(pretrained=True, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
       #load part of the weights from model zoo
       pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
       model_dict = model.state_dict()
       # 1. filter out unnecessary keys
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       # 2. overwrite entries in the existing state dict
       model_dict.update(pretrained_dict) 
       # 3. load the new state dict
       model.load_state_dict(model_dict)
    return model

