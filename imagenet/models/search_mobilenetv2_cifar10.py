"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from .layers import aw_noise_Conv2d, aw_noise_Linear, aw_BatchNorm2d
__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, ch_width, ch_index, stride):
    return nn.Sequential(
        aw_noise_Conv2d(inp, ch_width, 3, stride, 1, bias=False, channel_index_out=ch_index),
        aw_BatchNorm2d(ch_width, channel_index=ch_index),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, ch_in, ch_out, index_in, index_out):
    return nn.Sequential(
        aw_noise_Conv2d(inp, oup, 1, 1, 0, bias=False,
                        channel_index_in=index_in,
                        channel_index_out=index_out),
        aw_BatchNorm2d(oup, channel_index=index_out),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ch_in, ch_out, index_in, index_out, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        ch_hidden = round(ch_in* expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                aw_noise_Conv2d(ch_hidden, ch_hidden, 3, stride, 1, groups=ch_hidden, bias=False,
                                channel_index_out=index_in),
                aw_BatchNorm2d(ch_hidden),
                nn.ReLU6(inplace=True),
                # pw-linear
                aw_noise_Conv2d(ch_hidden, ch_out, 1, 1, 0, bias=False),
                aw_BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                aw_noise_Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                aw_BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                aw_noise_Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                aw_BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                aw_noise_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                aw_BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],
            [6,  32, 3, 1],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.ch_width = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 384, 96, 96, 96, 160, 160, 160, 320, 1280]

        self.ch_index = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]


        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, self.ch_width[0], self.ch_index[0], 2)]
        # building inverted residual blocks
        block = InvertedResidual
        count = 0
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, self.ch_width[count], self.ch_width[count+1], self.ch_index[count], self.ch_index[count+1],
                                s if i == 0 else 1, t))
                input_channel = output_channel
                count += 1
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        # self.conv = conv_1x1_bn(input_channel, output_channel, self.ch_width[18], self.ch_width[19], self.ch_index[18], self.ch_index[19])
        self.features.append(conv_1x1_bn(input_channel, output_channel, self.ch_width[18], self.ch_width[19], self.ch_index[18], self.ch_index[19]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = aw_noise_Linear(output_channel, self.ch_width[19], self.ch_index[19], num_classes)
        self.classifier = aw_noise_Linear(self.ch_width[-1], num_classes, channel_index=self.ch_index[-1])
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def search_mobilenetv2_cifar(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)
