import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

from .layers import nas_BatchNorm2d, nas_DownsampleA
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d, nas_noise_Conv2d1, NasBatchNorm2d, NasBatchNorm2d1
from .layers import aw_noise_Conv2d, aw_noise_Linear, aw_BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class DownsampleA(nn.Module):

  def __init__(self, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return aw_noise_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_mid, ch_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nas_noise_Conv2d(ch_in,
                                ch_mid,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn1 = NasBatchNorm2d(ch_mid)

        self.conv2 = nas_noise_Conv2d(ch_mid,
                                ch_out,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn2 = NasBatchNorm2d(ch_out)

        self.downsample = downsample
        
        self.conv_r = nas_noise_Conv2d(ch_in, ch_out, 1,  1, 0, bias=False)
        self.bn_r = NasBatchNorm2d(ch_out)

    def forward(self, x):
        residual = self.conv_r(x)
        residual = self.bn_r(residual)
  
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
     
        if self.downsample is not None:
            residual = self.downsample(residual)
     
        out += residual
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = noise_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = noise_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = noise_Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.ch_width = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        # self.ch_index = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]

        self.idx = 0
        self.conv1 = nas_noise_Conv2d(3, self.ch_width[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = NasBatchNorm2d(self.ch_width[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], self.ch_width[:5])
        self.layer2 = self._make_layer(block, 128, layers[1], self.ch_width[4:9],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], self.ch_width[8:13], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], self.ch_width[12:17], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nas_noise_Linear(self.ch_width[16] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, ch_width, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
              downsample = DownsampleA(stride)

            
        # else:

        #     downsample = nas_DownsampleA(ch_width[0],
        #                             ch_width[2], stride)
        
        layers = []
        layers.append(block(ch_width[0], ch_width[1], ch_width[2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(ch_width[i*2], ch_width[i*2+1],ch_width[i*2+2]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def update_model(self):

        l = 0
        for name, m in self.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if 'conv_r' not in name:
                    if l > 0:
                        m.in_channels = self.ch_width[l-1]
              
                    m.out_channels = self.ch_width[l]
                   

                    l += 1
                if 'conv_r' in name:
                    # print(l)
                    m.in_channels = self.ch_width[l-3]
                    m.out_channels = self.ch_width[l-1]

            
            if isinstance(m, NasBatchNorm2d):
                if 'bn_r' not in name:
                    m.num_features = self.ch_width[l-1]
                    m.update_idx(self.idx)

                if 'bn_r' in name:
                    m.num_features = self.ch_width[l-1]
                    m.update_idx(self.idx)

            if isinstance(m, nn.Linear):
                m.in_features = self.ch_width[-1]
                
    


def ens_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    #load part of the weights from model zoo
    # pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    # # if pretrained:
    # #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def noise_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model