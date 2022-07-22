import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import models
import time
from cal_flops import print_model_parm_flops


def subnets_linear(arch_list, acc, path):
    acc_index = []
    arch_iter_84 = []
    # acc_index_pni = [0,10,23,30,40, 49, 59, 68,79,87,95,105,114]
    # acc_index_pni_decend = [0, 10, 18, 28, 39, 49, 57, 66, 76, 89, 97,106,114]
    # acc_index_l1 = [0, 10, 23, 32,39, 47,57,66,77,86,  94,106, 114]
    for i in range(len(arch_list)):
        if i < 84:
            print(i, acc[i][2], arch_list[i])
            arch_iter_84.append(arch_list[i])
        # if i in acc_index_l1:
        #     print(i)
        #     arch_iter_13.append(arch_list[i])

    np.save(path+'/arch_iter_84', arch_iter_84)   
    print(arch_list[0])

def test_params(_Num, net):
    
    params_list = []
    acc_list = []
    flops_list = []
    # for i in np.arange(0,58,3):
    for i in np.arange(_Num):
        t0 = time.time()
        print('==================')
        print(i)
        # 37 457.183592
        # 20 998.112232
        # if i == 0 or 57:
        num_arch = int(i)
        curr_arch = []
        for j in range(len(arch_list[0])):
            curr_arch.append(int(arch_list[num_arch][j]))
        net.apply(
        lambda m: setattr(m, 'ch_width', curr_arch))
        net.apply(
        lambda m: setattr(m, 'idx', 0))
        if i == _Num-1:
            net.apply(
            lambda m: setattr(m, 'idx', 3))

        net.update_model()
        print(net.ch_width)

        M_flops, M_params = print_model_parm_flops(net) 
        
        # print('evaluate one time:{}'.format(time.time()-t0))
        print('M_flops:{}, M_params:{}'.format(M_flops, M_params))

def test_usnn(_Num, net):

    width_list = np.arange(1.0, 0.3, -0.025)
    for i in range(len(width_list)):
    # for i in [1.0, 0.75,0.5,0.25]:
        print('==========')
        print(i, width_list[i])
        
        curr_arch = []
        base = arch_list[0][0]

        # for j in range(len(arch_list[0])):
        #     curr_arch.append(int(arch_list[57][j]*i))
        for j in range(len(arch_list[0])):
            curr_arch.append(int(arch_list[0][j] * width_list[i]))

            # curr_arch.append(int(base * width_list[i])*int(arch_list[0][j]/base))
            # curr_arch.append(int(base * i)*int(arch_list[0][j]/base))

        net.apply(
        lambda m: setattr(m, 'ch_width', curr_arch))
        net.apply(
            lambda m: setattr(m, 'idx', 0))
        net.update_model()     
        M_flops, M_params = print_model_parm_flops(net) 

        print(net.ch_width)

        print('M_flops:{}, M_params:{}'.format(M_flops, M_params))

      
path = './save/subnet_search/mobilenetv1/1_l1_ft20_group4'
arch_list = np.load(path+'/arch_iter.npy')
acc = np.load(path + '/channel_layer_acc_list.npy')

for i in range(len(arch_list)):
    print(i, arch_list[i])
print(len(arch_list))

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
# curr_arch = []
# for i in range(len(arch_list[0])):
#     curr_arch.append(int(arch_list[0][i]))

# net = models.__dict__['nas_noise_mobilenetv2'](curr_arch)
# net.cuda()  
# _Num = len(arch_list)
# test_params(_Num, net)
# subnets_linear(arch_list, acc, path)
