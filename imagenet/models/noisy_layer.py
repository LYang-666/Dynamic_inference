import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np


class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='channelwise', w_noise=False):
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        
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
        
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.linear(input, noise_weight, self.bias)
        
        return output 

class nas_noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='channelwise', w_noise=False):
        super(nas_noise_Linear, self).__init__(in_features, out_features, bias)
        
        self.pni = pni
        # if self.pni is 'layerwise':
        #     self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        # elif self.pni is 'channelwise':
        #     self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*0,
        #                                 requires_grad=True)
        # elif self.pni is 'elementwise':
        #     self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        
        # with torch.no_grad():
        #     std = self.weight[:self.out_features, :self.in_features].std().item()
        #     noise = self.weight[:self.out_features, :self.in_features].clone().normal_(0,std)
        noise_weight = self.weight[:self.out_features, :self.in_features] 
        # noise_weight = self.weight[:self.out_features, :self.in_features] + self.alpha_w[:self.out_features] * noise * self.w_noise

        bias = self.bias[:self.out_features]
        output = F.linear(input, noise_weight, bias)
        
        return output 

class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0,
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

class noise_Conv2d1(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=True):
        super(noise_Conv2d1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0,
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

class nas_noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=False):
        super(nas_noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        # if self.pni is 'layerwise':
        #     self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        # elif self.pni is 'channelwise':
        #     self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0,
        #                                 requires_grad = True)     
        # elif self.pni is 'elementwise':
        #     self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise    
        self.groups = groups

    def forward(self, input):

        # with torch.no_grad():
        #     std = self.weight[:self.out_channels, :self.in_channels].std().item()
        #     noise = self.weight[:self.out_channels, :self.in_channels].clone().normal_(0,std)
        weight = self.weight[:self.out_channels, :self.in_channels]
        if self.groups > 1: 
            self.groups = self.out_channels
        # print(self.out_channels)
        # print(weight.shape)
        # noise_weight = weight + self.alpha_w[:self.out_channels] * noise * self.w_noise

        # noise_weight = weight + self.alpha_w[:self.out_channels] * noise * self.w_noise
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = None
        output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output


class nas_noise_Conv2d1(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=False):
        super(nas_noise_Conv2d1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0,
                                        requires_grad = True)     
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise    


    def forward(self, input):

        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight[:self.out_channels, :self.in_channels].clone().normal_(0,std)
        weight = self.weight[:self.out_channels, :self.in_channels]
        # print(self.out_channels)
        # print(weight.shape)
        noise_weight = weight + self.alpha_w[:self.out_channels] * noise * self.w_noise
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output


class NasBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(NasBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # self.idx = 0
        # for tracking performance during training
        self.bn = nn.ModuleList(
            nn.BatchNorm2d(i, affine=False) for i in [self.num_features_max, int(self.num_features_max*0.25)]
        )
    def forward(self, input):
        c = self.num_features
        weight = self.weight
        bias = self.bias
        if self.idx in [0,3]:
            idx = self.idx
            if self.idx == 3:
                idx = 1
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps) 
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y

    def update_idx(self, idx):
        self.idx = idx
# class NasBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features):
#         super(NasBatchNorm2d, self).__init__(
#             num_features, affine=True, track_running_stats=False)
#         self.num_features_max = num_features
#         # for tracking performance during training
#         self.bn = nn.ModuleList(
#             nn.BatchNorm2d(i, affine=False) for i in [self.num_features_max, self.num_features_max, self.num_features_max]
#         )
#     def forward(self, input):
#         c = self.num_features
#         weight = self.weight
#         bias = self.bias
#         if self.idx in [0,1,2]:
#             idx = self.idx
#             # if self.idx ==3:
#             #     idx = 1
#             y = nn.functional.batch_norm(
#                 input,
#                 self.bn[idx].running_mean[:c],
#                 self.bn[idx].running_var[:c],
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps) 
#         else:
#             y = nn.functional.batch_norm(
#                 input,
#                 self.running_mean,
#                 self.running_var,
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps)
#         return y
        
#     def update_idx(self, idx):
#         self.idx = idx

# class NasBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features):
#         super(NasBatchNorm2d, self).__init__(
#             num_features, affine=True, track_running_stats=False)
#         self.num_features_max = num_features
#         # for tracking performance during training
#         self.bn = nn.ModuleList(
#             nn.BatchNorm2d(i, affine=False) for i in [self.num_features_max,  int(self.num_features_max*0.25)]
#         )
#     def forward(self, input):
#         c = self.num_features
#         weight = self.weight
#         bias = self.bias
#         if self.idx in [4,1]:
#             idx = self.idx
#             if self.idx ==1:
#                 idx = 1
#             if self.idx ==4:
#                 idx = 0
#             y = nn.functional.batch_norm(
#                 input,
#                 self.bn[idx].running_mean[:c],
#                 self.bn[idx].running_var[:c],
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps) 
#         else:
#             y = nn.functional.batch_norm(
#                 input,
#                 self.running_mean,
#                 self.running_var,
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps)
#         return y
        
#     def update_idx(self, idx):
#         self.idx = idx

class NasBatchNorm2d1(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(NasBatchNorm2d1, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList(
            nn.BatchNorm2d(i, affine=False) for i in [self.num_features_max, self.num_features_max]
        )
    def forward(self, input):
        c = self.num_features
        weight = self.weight
        bias = self.bias
        if self.idx in [4,1]:
            idx = self.idx
            if self.idx ==1:
                idx = 1
            if self.idx ==4:
                idx = 0
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps) 
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y
        
    def update_idx(self, idx):
        self.idx = idx

# class NasBatchNorm2d1(nn.BatchNorm2d):
#     def __init__(self, num_features):
#         super(NasBatchNorm2d1, self).__init__(
#             num_features, affine=True, track_running_stats=False)
#         self.num_features_max = num_features
#         # for tracking performance during training
#         self.bn = nn.ModuleList(
#             nn.BatchNorm2d(i, affine=False) for i in [self.num_features_max, self.num_features_max]
#         )
#     def forward(self, input):
#         c = self.num_features
#         weight = self.weight
#         bias = self.bias
#         if self.idx in [0,57]:
#             idx = self.idx
#             if self.idx ==57:
#                 idx = 1
#             y = nn.functional.batch_norm(
#                 input,
#                 self.bn[idx].running_mean[:c],
#                 self.bn[idx].running_var[:c],
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps) 
#         else:
#             y = nn.functional.batch_norm(
#                 input,
#                 self.running_mean,
#                 self.running_var,
#                 weight[:c],
#                 bias[:c],
#                 self.training,
#                 self.momentum,
#                 self.eps)
#         return y

#     def update_idx(self, idx):
#         self.idx = idx
# class adp_DownsampleA(nn.Module):
#     def __init__(self, nIn, nOut, stride, channel_index=[-1]):
#         super(aw_DownsampleA, self).__init__()

#         self.nIn = nIn
#         self.nOut = nOut
#         self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
        
#         # self._update_n_channels(channel_index)
        

#     def forward(self, x):
        
#         x = self.avg(x)
        
#         if self.nOut > self.nIn: 
#             pad = (0,0,0,0,0,self.nOut - self.nIn)
#             output = F.pad(x, pad)
#         else:
#             output = x[:, :self.nOut]

#         return output
    
#     def _update_n_channels(self, channel_index):
#         # for val in channel_ratio:
#         #     assert (val > 0) and (
#         #         val <= 1.), "channel_ratio is out of the (0,1) bound."
#         self.channel_index = channel_index
        
#         # feature/channels to be selected.
#         self.sel_nIn = self.nIn
#         self.sel_nOut = self.nOut