'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .noisy_layer import nas_noise_Linear,  nas_noise_Conv2d, nas_noise_Conv2d1, NasBatchNorm2d, NasBatchNorm2d1, noise_Conv2d, noise_Conv2d1, noise_Linear
from .layers import aw_Conv2d, aw_noise_Conv2d, aw_Linear, aw_noise_Linear, aw_BatchNorm2d, aw_DownsampleA


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, ch_in, ch_mid, ch_out, index_in, index_mid, index_out, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.in_planes = in_planes
        self.out_planes = out_planes

        planes = expansion * in_planes
        self.conv1 = aw_noise_Conv2d(ch_in, ch_mid, kernel_size=1, stride=1, padding=0, bias=False,
                                    channel_index_in=index_in,
                                    channel_index_out=index_mid)
        self.bn1 = aw_BatchNorm2d(ch_mid, channel_index=index_mid)
        self.conv2 = aw_noise_Conv2d(ch_mid, ch_mid, kernel_size=3, stride=stride, padding=1, groups=ch_mid, bias=False,
                                    channel_index_out=index_mid)
        self.bn2 = aw_BatchNorm2d(ch_mid, channel_index=index_mid)
        self.conv3 = aw_noise_Conv2d(ch_mid, ch_out, kernel_size=1, stride=1, padding=0, bias=False,
                                    channel_index_in=index_mid,
                                    channel_index_out=index_out)
        self.bn3 = aw_BatchNorm2d(ch_out, channel_index=index_out)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                aw_noise_Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False,
                                    channel_index_in=index_in,
                                    channel_index_out=index_out                ),
                aw_BatchNorm2d(ch_out, channel_index=index_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # if self.stride==1:
        #     print(out.size(), self.shortcut(x).size())
        out = out + self.shortcut(x) if self.stride==1 and self.in_planes != self.out_planes else out
         
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
       
        self.ch_width = [32, 32, 16, 96, 24, 144, 24, 144, 32, 192, 32, 192, 32, 192, 64, 384, 64, 384, 64, 384, 64, 384, 96, 576, 96, 576, 96, 576, 160, 960, 160, 960, 160, 960, 320, 1280]
        self.ch_index = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]

        self.idx = 0
        self.conv1 = aw_noise_Conv2d(3, self.ch_width[0], kernel_size=3, stride=1, padding=1, bias=False, channel_index_out=self.ch_index[0])
        self.bn1 = aw_BatchNorm2d(self.ch_width[0], self.ch_index[0])
        self.layers = self._make_layers(self.ch_width, self.ch_index, in_planes=32)
        self.conv2 = aw_noise_Conv2d(self.ch_width[34], self.ch_width[35], kernel_size=1, stride=1, padding=0, bias=False, channel_index_in=self.ch_index[34],
                                    channel_index_out=self.ch_index[35])
        self.bn2 = aw_BatchNorm2d(self.ch_width[35], self.ch_index[35])
        self.linear = aw_noise_Linear(self.ch_width[35], num_classes, channel_index=self.ch_index[-1])

    def _make_layers(self, ch_width, ch_index, in_planes):
        layers = []
        i = 0
        count = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            j = 0
            # print(strides)
            for stride in strides:
                # 0, 1, 2
                # 2,3,4
                # print(i, j, len(strides))
                # print(count)
                # print(i*(j*2+3),i*(j*2+3)+1, i*(j*2+3)+2)
                width = count * 2 
                layers.append(Block(in_planes, out_planes, ch_width[width], ch_width[width+1], ch_width[width+2],
                                ch_index[width], ch_index[width+1], ch_index[width+2], expansion, stride))

                # layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
                j+= 1
                count += 1
            i+=1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def update_model(self, layer):

        l = 0
        for name, m in self.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if 'shortcut' not in name and '.conv2' not in name:
                    if l > 0 and l==(layer+1):
                        m.in_channels = self.ch_width[l-1]
                        m.channel_index_in = self.ch_index[l-1]
                        m._update_n_channels(self.ch_index[l-1], in_ch=True)
                    if l == layer:
                        m.out_channels = self.ch_width[l]
                        m.channel_index_out = self.ch_index[l]
                        m._update_n_channels(self.ch_index[l], out_ch=True)                    

                    l += 1
                if 'shortcut' in name:
                    # print(l)
                    m.in_channels = self.ch_width[l-3]
                    m.out_channels = self.ch_width[l-1]
                    m.channel_index_in = self.ch_index[l-3]
                    m.channel_index_out = self.ch_index[l-1]
                    m._update_n_channels(self.ch_index[l-3], in_ch=True)
                    m._update_n_channels(self.ch_index[l-1], out_ch=True)

                if '.conv2' in name:
                    # print(l)
                    m.in_channels = self.ch_width[l-1]
                    m.out_channels = self.ch_width[l-1]
                    m.channel_index_in = self.ch_index[l-1]
                    m.channel_index_out = self.ch_index[l-1]
                    m._update_n_channels(self.ch_index[l-1], in_ch=True)
                    m._update_n_channels(self.ch_index[l-1], out_ch=True)

            if isinstance(m, nn.BatchNorm2d):
                if 'shortcut' not in name:
                    if l == (layer+1):
                        m.num_features = self.ch_width[l-1]
                        m.channel_index = self.ch_index[l-1]
                        m._update_n_channels(self.ch_index[l-1])  
                if 'shortcut' in name:
                    m.num_features = self.ch_width[l-1]
                    m.channel_index = self.ch_index[l-1]
                    m._update_n_channels(self.ch_index[l-1])  
            if isinstance(m, nn.Linear):
                m.in_features = self.ch_width[-1]
                m.channel_index = self.ch_index[-1]
                m._update_n_channels(self.ch_index[-1])    
                        


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

def search_noise_mobilenet(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  	Args:
    	num_classes (uint): number of classes
  	"""
    model = MobileNetV2(num_classes)
    return model