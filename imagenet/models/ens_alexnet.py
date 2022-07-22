import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.nn.init as init
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d,  NasBatchNorm2d,  noise_Conv2d, noise_Conv2d1, noise_Linear
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear, aw_BatchNorm2d, aw_DownsampleA


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.ch_width = [64, 192, 384, 256, 256, 4096, 4096]
        self.idx = 0
        self.features = nn.Sequential(
            nas_noise_Conv2d(3, self.ch_width[0], kernel_size=11, stride=4, padding=2),
            NasBatchNorm2d(self.ch_width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nas_noise_Conv2d(self.ch_width[0], self.ch_width[1], kernel_size=5, padding=2),
            NasBatchNorm2d(self.ch_width[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nas_noise_Conv2d(self.ch_width[1], self.ch_width[2], kernel_size=3, padding=1),
            NasBatchNorm2d(self.ch_width[2]),
            nn.ReLU(inplace=True),
            nas_noise_Conv2d(self.ch_width[2], self.ch_width[3], kernel_size=3, padding=1),
            NasBatchNorm2d(self.ch_width[3]),
            nn.ReLU(inplace=True),
            nas_noise_Conv2d(self.ch_width[3], self.ch_width[4], kernel_size=3, padding=1),
            NasBatchNorm2d(self.ch_width[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nas_noise_Linear(self.ch_width[4] * 6 * 6, self.ch_width[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nas_noise_Linear(self.ch_width[5], self.ch_width[6]),
            nn.ReLU(inplace=True),
            nas_noise_Linear(self.ch_width[6], num_classes),
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

                if l == 7:
                    m.in_features = self.ch_width[-1]
          
                else:
                    m.in_features = self.ch_width[l-1]
                    m.out_features = self.ch_width[l]
                    if l == 5:
                        m.in_features = self.ch_width[l-1]*6*6
                l+=1
            if isinstance(m, NasBatchNorm2d):

                m.num_features = self.ch_width[l-1]
                m.update_idx(self.idx)


def ens_alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

