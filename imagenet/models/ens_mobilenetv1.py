'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear, aw_BatchNorm2d, aw_DownsampleA
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d, nas_noise_Conv2d1, NasBatchNorm2d, NasBatchNorm2d1, noise_Conv2d, noise_Conv2d1, noise_Linear
import math

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, ch_in, ch_out, stride=1):
        super(Block, self).__init__()
        self.conv1 = nas_noise_Conv2d(ch_in, ch_in, kernel_size=3, stride=stride, padding=1, groups=ch_in, bias=False)
        self.bn1 = NasBatchNorm2d(ch_in)
        self.conv2 = nas_noise_Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = NasBatchNorm2d(ch_out)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        self.ch_width = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

        self.ch_index = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
        self.idx = 0
        self.conv1 = nas_noise_Conv2d(3, self.ch_width[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = NasBatchNorm2d(self.ch_width[0])
        self.layers = self._make_layers(self.ch_width, in_planes=32)
        self.linear = nas_noise_Linear(self.ch_width[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layers(self, ch_width, in_planes):
        layers = []
        i = 0
        # 0  0,1
        # 1  1,2
        for x in self.cfg:
            
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, ch_width[i], ch_width[i+1], stride))
            in_planes = out_planes
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def update_model(self):

        l = 0
        for name, m in self.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if '.conv1' not in name:
                    if l > 0 :
                        m.in_channels = self.ch_width[l-1]
                   
                    m.out_channels = self.ch_width[l]
                             
                    l += 1

                if '.conv1' in name:
                    # print(l)
                   
                    m.in_channels = self.ch_width[l-1]
                    m.out_channels = self.ch_width[l-1]
        
                        # print(m.out_channels)

            if isinstance(m, NasBatchNorm2d):

                m.num_features = self.ch_width[l-1]
                m.update_idx(self.idx)

            if isinstance(m, nn.Linear):
                m.in_features = self.ch_width[-1]
    

def ens_mobilenetv1(arch, num_classes=1000):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  	Args:
    	num_classes (uint): number of classes
  	"""
    model = MobileNet(num_classes)
    return model



if __name__ == '__main__':
    net = MobileNet()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

# test()