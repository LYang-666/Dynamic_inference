import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


def make_divisible(v, divisor=1, min_value=1):
    """
    v: base width to be adjusted
    divisor: width divisor, minimal adjustable width
    min_value: minimal width
    
    Floor the channel number approximately as:
        round(v / d) * d 
    
    forked from slim:
    https://github.com/tensorflow/models/blob/0344c5503ee55e24f0de7f37336a6e08f10976fd/research/slim/nets/mobilenet/mobilenet.py#L62-L69
    https://github.com/JiahuiYu/slimmable_networks/blob/d211f5bc5b88918a25bdda9b61cfc9a4936b2a62/models/slimmable_ops.py#L76
    """
    if min_value is None:
        min_value = divisor
    # new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    new_v = max(min_value, math.floor(v/divisor) * divisor)
    # # Make sure that round down does not go down by more than 10%.
    # if new_v < 0.9 * v:
        # new_v += divisor
    return new_v


def round_channel(n_channels, width_ratio):
    '''round up the channel to integer.
    '''
    return math.ceil(n_channels * width_ratio)



class aw_Conv2d(nn.Conv2d):
    ''' 
    This function is an re-implementation Conv2d with adjustable width.

    Args:
        channel_ratio (list): [input channel ratio, out channel ratio]
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 channel_index=[-1]):
        super(aw_Conv2d, self).__init__(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias)

        self.channel_index = channel_index
        self.max_in_ch = self.in_channels
        self.max_out_ch = self.out_channels
        self._update_n_channels(channel_index=self.channel_index)

    def forward(self, input):
        # return the identity if the #channels are rounded to 0.
        if len(self.sel_in_channels) == 0 or len(self.sel_out_channels) == 0:
            output = input
        else:
            if self.bias is None:
                bias = None
            else:
                bias = self.bias[self.sel_out_channels]

            output = F.conv2d(
                input,
                self.weight[self.sel_out_channels, self.sel_in_channels],
                bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def _update_n_channels(self, channel_index, out_ch=False, in_ch=False):

        # for val in channel_ratio:
        #     assert (val > 0.) and (
        #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        # self.channel_ratio = channel_ratio
        self.channel_index = channel_index
        self.sel_in_channels = [i for i in range(self.max_in_ch)]
        self.sel_out_channels = [i for i in range(self.max_out_ch)]
        if out_ch:
            # the output channels index without the current pruned channel_index
            self.sel_out_channels = [i for i in range(self.max_out_ch) if i is not channel_index]
        if in_ch:
            # the input channels index without the current pruned channel_index
            self.sel_in_channels = [i for i in range(self.max_in_ch) if i is not channel_index]                




class aw_noise_Conv2d(nn.Conv2d):
    ''' 
    This function is an re-implementation Conv2d with adjustable width.

    Args:
        channel_ratio (list): [input channel ratio, out channel ratio]
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 channel_index_in=[-1],
                 channel_index_out=[-1],
                 pni='channelwise', 
                 w_noise=True):
        super(aw_noise_Conv2d, self).__init__(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias)

        self.channel_index_in = channel_index_in
        self.channel_index_out = channel_index_out

        self.max_in_ch = self.in_channels
        self.max_out_ch = self.out_channels
        self._update_n_channels(channel_index=self.channel_index_out)
        self.sel_in_channels = [i for i in range(self.max_in_ch)]
        self.sel_out_channels = [i for i in range(self.max_out_ch)]

        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.,
                                        requires_grad = True)     
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise   

    def forward(self, input):
        # return the identity if the #channels are rounded to 0.
        if len(self.sel_in_channels) == 0 or len(self.sel_out_channels) == 0:
            output = input
        else:
            if self.bias is None:
                bias = None
            else:
                bias = self.bias[self.sel_out_channels]

            weight = self.weight[self.sel_out_channels, :]
            if self.groups == 1:
                weight = weight[:, self.sel_in_channels]

            with torch.no_grad():
                std = weight.std().item()
                noise = weight.clone().normal_(0,std)   
            if self.training:
                noise_weight = weight + self.alpha_w[self.sel_out_channels] * noise * self.w_noise
            else:
                noise_weight = weight
            output = F.conv2d(
                input,
                noise_weight,
                bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def _update_n_channels(self, channel_index, out_ch=False, in_ch=False):

        # for val in channel_ratio:
        #     assert (val > 0.) and (
        #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        # self.channel_ratio = channel_ratio
        self.channel_index = channel_index

        if out_ch:
            # the output channels index without the current pruned channel_index
            self.sel_out_channels = [i for i in range(self.max_out_ch) if i not in self.channel_index]
            if self.groups > 1: 
                self.groups = len(self.sel_out_channels)
        if in_ch:
            # the input channels index without the current pruned channel_index
            self.sel_in_channels = [i for i in range(self.max_in_ch) if i not in self.channel_index]    


class aw_Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 channel_index=-1):
        super(aw_Linear, self).__init__(in_features, out_features, bias=bias)

        self.max_in_ch = self.in_features

        self.channel_index = channel_index
        self._update_n_channels(channel_index=self.channel_index)

    def forward(self, input):
        if self.sel_in_features == 0 or self.sel_out_features == 0:
            output = input
        else:
            if self.bias is None:
                bias = None
            else:
                bias = self.bias[:self.sel_out_features]

            output = F.linear(
                input,
                self.weight[:self.sel_out_features, self.sel_in_features],
                # self.weight,
                bias)

        return output

    def _update_n_channels(self, channel_index):
        # for val in channel_ratio:
        #     assert (val > 0) and (
        #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        self.channel_index = channel_index
   
        self.sel_in_features = [i for i in range(self.in_features) if i not in self.channel_index]
        self.sel_out_features = self.out_features

class aw_noise_Linear1(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 channel_index_in=[-1],
                 channel_index_out=[-1],
                 pni='channelwise', 
                 w_noise=True):
        super(aw_noise_Linear1, self).__init__(in_features, out_features, bias=bias)

        # self.max_in_ch = self.in_features
        self.channel_index_in = channel_index_in
        self.channel_index_out = channel_index_out

        self.max_in_ch = self.in_features
        self.max_out_ch = self.out_features

        # self.channel_index = channel_index
        self._update_n_channels(channel_index=self.channel_index_out)
        self.sel_in_features = [i for i in range(self.max_in_ch)]
        self.sel_out_features = [i for i in range(self.max_out_ch)]
        
        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*0,
                                        requires_grad=True)
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        if len(self.sel_in_features) == 0 or len(self.sel_out_features) == 0:
            output = input
        else:
            if self.bias is None:
                bias = None
            else:
                bias = self.bias[self.sel_out_features]
            weight = self.weight[self.sel_out_features, :]
            weight = weight[:, self.sel_in_features]           
            with torch.no_grad():
                std = self.weight.std().item()
                noise = weight.clone().normal_(0,std)
            if self.training:
                noise_weight = weight + self.alpha_w[self.sel_out_features] * noise * self.w_noise
            else:
                noise_weight = weight
            output = F.linear(
                input,
                noise_weight,
                bias)

        return output

    def _update_n_channels(self, channel_index, out_ch=False, in_ch=False):
        # for val in channel_ratio:
        #     assert (val > 0) and (
        #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        self.channel_index = channel_index
   
        # self.sel_in_features = [i for i in range(self.max_in_ch) if i not in self.channel_index]
        # self.sel_out_features = self.out_features

        if out_ch:
            # the output channels index without the current pruned channel_index
            self.sel_out_features = [i for i in range(self.max_out_ch) if i not in self.channel_index]
  
        if in_ch:
            # the input channels index without the current pruned channel_index
            self.sel_in_features = [i for i in range(self.max_in_ch) if i not in self.channel_index]    



class aw_noise_Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 channel_index=[-1],
                 pni='channelwise', 
                 w_noise=True):
        super(aw_noise_Linear, self).__init__(in_features, out_features, bias=bias)

        self.max_in_ch = self.in_features

        self.channel_index = channel_index
        self._update_n_channels(channel_index=self.channel_index)
        
        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*0.,
                                        requires_grad=True)
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        if self.sel_in_features == 0 or self.sel_out_features == 0:
            output = input
        else:
            if self.bias is None:
                bias = None
            else:
                bias = self.bias[:self.sel_out_features]
            
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight[:self.sel_out_features, self.sel_in_features].clone().normal_(0,std)

            if self.training:
                noise_weight = self.weight[:self.sel_out_features, self.sel_in_features] + self.alpha_w[:self.sel_out_features] * noise * self.w_noise
            else:
                noise_weight = self.weight[:self.sel_out_features, self.sel_in_features]
            output = F.linear(
                input,
                noise_weight,
                bias)

        return output

    def _update_n_channels(self, channel_index):
        # for val in channel_ratio:
        #     assert (val > 0) and (
        #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        self.channel_index = channel_index
        self.sel_in_features = [i for i in range(self.max_in_ch) if i not in self.channel_index]
        self.sel_out_features = self.out_features


class aw_BatchNorm2d(nn.BatchNorm2d):
    '''
    Batch Normlaization-2d with the adjustable width (channel/features)
    '''
    def __init__(self, num_features, channel_index=[-1]):
        super(aw_BatchNorm2d, self).__init__(num_features)
        self.max_num_feat = self.num_features
        self._update_n_channels(channel_index)


    def forward(self, input):
        # self._check_input_dim(input)

        # exponential_average_factor = 0.0

        # if self.training and self.track_running_stats:
        #     self.num_batches_tracked += 1
        #     if self.momentum is None:  # use cumulative moving average
        #         exponential_average_factor = 1.0 / self.num_batches_tracked.item(
        #         )
        #     else:  # use exponential moving average
        #         exponential_average_factor = self.momentum

        return F.batch_norm(input, self.running_mean[self.sel_num_features],
                            self.running_var[self.sel_num_features],
                            self.weight[self.sel_num_features],
                            self.bias[self.sel_num_features], self.training,
                            self.momentum, self.eps)

    def _update_n_channels(self, channel_index):
      
        self.channel_index = channel_index
        
        self.sel_num_features = [i for i in range(self.max_num_feat) if i not in self.channel_index]
            

# class aw_BatchNorm2d(nn.BatchNorm2d):
#     '''
#     Batch Normlaization-2d with the adjustable width (channel/features)
#     '''
#     def __init__(self, num_features, channel_index=[-1]):
#         super(aw_BatchNorm2d, self).__init__(num_features)
#         self.max_num_feat = self.num_features
#         self._update_n_channels(channel_index)


#     def forward(self, input):
#         # self._check_input_dim(input)

#         # exponential_average_factor = 0.0

#         # if self.training and self.track_running_stats:
#         #     self.num_batches_tracked += 1
#         #     if self.momentum is None:  # use cumulative moving average
#         #         exponential_average_factor = 1.0 / self.num_batches_tracked.item(
#         #         )
#         #     else:  # use exponential moving average
#         #         exponential_average_factor = self.momentum

#         return F.batch_norm(input, self.running_mean[:self.num_features],
#                             self.running_var[:self.num_features],
#                             self.weight[:self.num_features],
#                             self.bias[:self.num_features], self.training,
#                             self.momentum, self.eps)

#     def _update_n_channels(self, channel_index):
      
#         self.channel_index = channel_index
        
#         self.sel_num_features = [i for i in range(self.max_num_feat) if i not in self.channel_index]

class nas_BatchNorm2d(nn.BatchNorm2d):
    '''
    Batch Normlaization-2d with the adjustable width (channel/features)
    '''
    def __init__(self, num_features):
        super(nas_BatchNorm2d, self).__init__(num_features)
        # self.max_num_feat = self.num_features
        # self._update_n_channels(channel_index)


    def forward(self, input):
        # self._check_input_dim(input)

        # exponential_average_factor = 0.0

        # if self.training and self.track_running_stats:
        #     self.num_batches_tracked += 1
        #     if self.momentum is None:  # use cumulative moving average
        #         exponential_average_factor = 1.0 / self.num_batches_tracked.item(
        #         )
        #     else:  # use exponential moving average
        #         exponential_average_factor = self.momentum
   
        return F.batch_norm(input, self.running_mean[:self.num_features],
                            self.running_var[:self.num_features],
                            self.weight[:self.num_features],
                            self.bias[:self.num_features], self.training,
                            self.momentum, self.eps)


class nas_DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(nas_DownsampleA, self).__init__()

        self.max_out = nOut
        self.nIn = nIn
        self.nOut = nOut
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        
        # self._update_n_channels(channel_index)
        

    def forward(self, x):
        
        x = self.avg(x)
        
        if self.nOut > self.nIn: 
            pad = (0,0,0,0,0,self.nOut - self.nIn)
            output = F.pad(x, pad)
        else:
            output = x[:, :self.nOut]

        # mask = torch.ones(output.shape[1]).cuda()
        # if len(self.channel_index) > 1:
        #     mask[self.channel_index[1:]] = 0
        #     # print(mask)

        # mask = mask.view(len(mask), 1).expand(-1, output.size(0)).view(len(mask), output.size(0), 1, 1).transpose(1,0)

        return output


class aw_DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride, channel_index=[-1]):
        super(aw_DownsampleA, self).__init__()

        self.max_out = nOut
        self.nIn = nIn
        self.nOut = nOut
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        
        self._update_n_channels(channel_index)
        

    def forward(self, x):
        
        x = self.avg(x)
        
        if self.nOut > self.nIn: 
            pad = (0,0,0,0,0,self.nOut - self.nIn)
            output = F.pad(x, pad)
            
        else:
            output = x[:, self.sel_nOut]

        # mask = torch.ones(output.shape[1]).cuda()
        # if len(self.channel_index) > 1:
        #     mask[self.channel_index[1:]] = 0
        #     # print(mask)

        # mask = mask.view(len(mask), 1).expand(-1, output.size(0)).view(len(mask), output.size(0), 1, 1).transpose(1,0)

        return output
    
    def _update_n_channels(self, channel_index):
    #     # for val in channel_ratio:
    #     #     assert (val > 0) and (
    #     #         val <= 1.), "channel_ratio is out of the (0,1) bound."
        self.channel_index = channel_index
        
        # feature/channels to be selected.
        self.sel_nIn = self.nIn

        self.sel_nOut =  [i for i in range(self.max_out) if i not in self.channel_index]

# class aw_DownsampleA(nn.Module):
#     def __init__(self, nIn, nOut, stride, channel_index=[-1]):
#         super(aw_DownsampleA, self).__init__()

#         # self.max_out = nOut
#         self.nIn = nIn
#         self.nOut = nOut
#         self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        
#         self._update_n_channels(channel_index)
        

#     def forward(self, x):
        
#         x = self.avg(x)
        
#         if self.nOut > self.nIn: 
#             pad = (0,0,0,0,0,self.nOut - self.nIn)
#             output = F.pad(x, pad)
#         else:
#             output = x[:, :self.nOut]

#         # mask = torch.ones(output.shape[1]).cuda()
#         # if len(self.channel_index) > 1:
#         #     mask[self.channel_index[1:]] = 0
#         #     # print(mask)

#         # mask = mask.view(len(mask), 1).expand(-1, output.size(0)).view(len(mask), output.size(0), 1, 1).transpose(1,0)

#         return output
    
#     def _update_n_channels(self, channel_index):
#     #     # for val in channel_ratio:
#     #     #     assert (val > 0) and (
#     #     #         val <= 1.), "channel_ratio is out of the (0,1) bound."
#         self.channel_index = channel_index
        
#         # feature/channels to be selected.
#         # self.sel_nIn = self.nIn
#         # self.sel_nOut =  self.nIn

#         # self.sel_nOut =  [i for i in range(self.max_out) if i not in self.channel_index]

           



class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.25,
                                        requires_grad = True)     
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise    


    def forward(self, input):

        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output


###############
# testbench
# > pytest *.py -v -s

# import pytest
# import timeit


# # https://stackoverflow.com/questions/43071005/check-if-expection-is-raised-with-pytest
# # http://doc.pytest.org/en/latest/assert.html
# def test_aw_Conv2d():
#     ''' check the functionality of aw_Conv2d'''
#     with pytest.raises(AssertionError, match=r"channel_ratio .*"):
#         m = aw_Conv2d(1, 3, 3, channel_ratio=[-1, 1 / 3])

#     with pytest.raises(AssertionError, match=r"channel_ratio .*"):
#         m = aw_Conv2d(1, 3, 3, channel_ratio=[1 / 3, 2])

#     m = aw_Conv2d(10, 20, 3, channel_ratio=[1 / 3, 2 / 3])
#     print('\n', m.channel_ratio, m.sel_in_channels, m.sel_out_channels)
#     print('Original weight size:', m.weight.size())
#     print('selected weight size:',
#           m.weight[:m.sel_out_channels, :m.sel_in_channels].size())

#     # check the dimension correctness with dummpy input and module
#     x = torch.rand(3, m.sel_in_channels, 13, 13)
#     y = m(x)

#     # use a different size of input
#     with pytest.raises(RuntimeError,
#                        match=r".* but got {} channels instead".format(
#                            2 * m.sel_in_channels)):
#         x = torch.rand(3, 2 * m.sel_in_channels, 13, 13)
#         y = m(x)


# def test_aw_Linear():
#     with pytest.raises(AssertionError, match=r"channel_ratio .*"):
#         m = aw_Linear(10, 10, channel_ratio=[-1, 0.6])

#     with pytest.raises(AssertionError, match=r"channel_ratio .*"):
#         m = aw_Linear(10, 10, channel_ratio=[.2, 2])

#     m = aw_Linear(10, 10, channel_ratio=[.4, .6])
#     print('\n', m.channel_ratio, m.sel_in_features, m.sel_out_features)
#     print('Original weight size:', m.weight.size())
#     print('selected weight size:',
#           m.weight[:m.sel_out_features, :m.sel_in_features].size())

#     # check the dimension correctness with dummpy input and module
#     x = torch.rand(3, m.sel_in_features)
#     y = m(x)
#     print(y.size())

#     # use a different size of input
#     with pytest.raises(RuntimeError, match=r"size mismatch, .*"):
#         x = torch.rand(3, 2 * m.sel_in_features)
#         y = m(x)


# def test_aw_BatchNorm2d():
#     m = aw_BatchNorm2d(10, channel_ratio=1)
#     print('\n', 'selected features:', m.sel_num_features)
#     print('init running mean:', m.running_mean)
#     print('init running var', m.running_var)

#     # test-case 1:
#     channel_ratio = 0.3
#     m._update_n_channels(channel_ratio=channel_ratio)
#     print('\n', 'selected features:', m.sel_num_features)
#     x = torch.rand(5, m.sel_num_features, 24, 24)
#     y = m(x)
#     print('one-iter running mean:', m.running_mean)
#     print('one-iter running var:', m.running_var)
#     assert m.running_mean[m.sel_num_features:].sum() < 1e-5 \
#         and m.running_mean[m.sel_num_features-1:].sum() > 1e-5, 'Batchnorm is not properly updated!'

#     # test-case 2
#     channel_ratio = 0.7
#     m._update_n_channels(channel_ratio=channel_ratio)
#     print('\n', 'selected features:', m.sel_num_features)
#     x = torch.rand(5, m.sel_num_features, 24, 24)
#     y = m(x)
#     print('one-iter running mean:', m.running_mean)
#     print('one-iter running var:', m.running_var)
#     assert m.running_mean[m.sel_num_features:].sum() < 1e-5 \
#         and m.running_mean[m.sel_num_features-1:].sum() > 1e-5, 'Batchnorm is not properly updated!'


# #######################
# # benchmark


# def benchmark_speed_aw_Conv2d(ratio=[0.5, 0.5], running_times=1):
#     """
#     https://stackoverflow.com/questions/44677606/how-to-measure-the-speed-of-a-python-function
#     https://docs.python.org/3/library/timeit.html
#     """
#     print("current ratio:", ratio)
#     print(
#         "inference speed:",
#         timeit.timeit("y=m(x)",
#                       setup="from __main__ import aw_Conv2d; import torch;\
#     m = aw_Conv2d(10, 20, 3, channel_ratio={});\
#     x = torch.rand(3, m.sel_in_channels, 24, 24)".format(ratio),
#                       number=running_times))


# def benchmark_speed_aw_Linear(ratio=[.5, .5], running_times=1):
#     '''similar as the Conv2d function above
#     '''

#     print("current ratio:", ratio)
#     print(
#         "inference speed:",
#         timeit.timeit("y=m(x)",
#                       setup="from __main__ import aw_Linear; import torch;\
#                         m = aw_Linear(300, 300, channel_ratio={});\
#                         x = torch.rand(3, m.sel_in_features)".format(ratio),
#                       number=running_times))


if __name__ == '__main__':
    # benchmark the speed
    ratio_list = [[1, 1], [0.75, 0.75], [0.5, 0.5], [0.25, 0.25]]

    print('> Benchmark the speed for aw_Conv2d: ')
    for ratio in ratio_list:
        print('-' * 50)
        benchmark_speed_aw_Conv2d(ratio, running_times=1000)

    print('\n', '> Benchmark the speed for aw_Linear: ')
    for ratio in ratio_list:
        print('-' * 50)
        benchmark_speed_aw_Linear(ratio, running_times=1000)
