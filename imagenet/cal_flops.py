#coding:utf8
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import numpy as np

def print_model_parm_flops(one_shot_model):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    multiply_adds = False
    list_conv=[]
    list_conv_params=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        mask = torch.ones(self.weight.size(0))
        count_zero = self.weight.data.detach().sum((1,2,3))

        num_zero = len(mask[count_zero==0])
        # print(num_zero)
        # print(output_channels)       
        params = (output_channels) * (kernel_ops + bias_ops)
        # params = (output_channels-num_zero) * (input_channels) * self.kernel_size[0] * self.kernel_size[1]

        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)
        list_conv_params.append(params)

    list_linear=[]
    list_linear_params=[]
    def linear_hook(self, input, output):


        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        params = weight_ops + bias_ops
        list_linear.append(flops)
        list_linear_params.append(params)
    list_bn=[]
    list_bn_params=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
        list_bn_params.append(input[0].nelement())
    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        # childrens = list(net.children())
        # if not childrens:
        #     if isinstance(net, torch.nn.Conv2d):
        #         net.register_forward_hook(conv_hook)
        #     if isinstance(net, torch.nn.Linear):
        #         net.register_forward_hook(linear_hook)
        #     if isinstance(net, torch.nn.BatchNorm2d):
        #         net.register_forward_hook(bn_hook)
        #     if isinstance(net, torch.nn.ReLU):
        #         net.register_forward_hook(relu_hook)
        #     if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
        #         net.register_forward_hook(pooling_hook)
        #     return
        # for c in childrens:
        #     foo(c)
    
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                handle1 = m.register_forward_hook(conv_hook)
            if isinstance(m, torch.nn.Linear):
                handle2 =m.register_forward_hook(linear_hook)
            if isinstance(m, torch.nn.BatchNorm2d):
                handle3 =m.register_forward_hook(bn_hook)
            if isinstance(m, torch.nn.ReLU):
                handle4 =m.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)       
        # return handle1,handle2,handle3

    foo(one_shot_model)
    model = one_shot_model

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    input = Variable(torch.rand(3,224,224).unsqueeze(0), requires_grad = False).cuda()
    out = one_shot_model(input)
    # print(len(list_conv))
    # print(len(list_linear))
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total_params = (sum(list_conv_params) + sum(list_linear_params) + sum(list_bn_params))
    M_flops = total_flops / 1e6
    #print('  + Number of FLOPs: %.2fM' % (M_flops))
    M_params = total_params / 1e6
    # handle1.remove()
    # handle2.remove()
    # handle3.remove()

    return M_flops, M_params







