import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
net = models.__dict__['search_mobilenetv1']()
a = []
b = []
i = 0
# print(net)
# for name, m in net.named_modules():
#     if isinstance(m, nn.Conv2d):
    
#         print(name, m.weight.size())
        # print(m.alpha_w.size())
        # if 'features.0.0' in name or '1.conv.3' in name or 'conv.6' in name or name == 'conv.0':

        #     # print(m.out_channels)
        #     print(name)
        #     print(i, m.out_channels)

        #     a.append(m.out_channels)
        #     b.append([-1])
        #     i+=1
# print(a)
# print(b)

# if 250 in np.arange(200, 300):
#     print('1')
a = [0,1,2,3]
b = []
for i in a:
    print(i)
    for j in np.arange(i*4, i*4+4):
        b.append(j)
print(b)
# b = [i for i in np.arange(200, 300)]
# a = torch.randn(5)
# print(a)
# b = a[[0,1,3,4]]
# print(b)
# a = torch.ones(5, 5)

# b= a[[0,1,2,3,4], [0,1,2,3,4]]
# print(b)
# # b = b[:,[1,2]]
# # print(b)
# a = [i for i in range(10) if i not in [1]]
# print(a)

# # a = torch.randn(6)
# # print(a)
# # b = a[[0,1,2,3,4,5]]
# # print(b)
# i = 0
# j = 0
# k = 1
# if j == 0:
#     i+=1
# if k ==1:
#     i+=1
# print(i)

# mask = torch.ones(res.shape[1]).cuda()
# channels = res.data.detach().sum((0,2,3))
# mask[channels==0] = 0
# for i in range(channels.size(0)):
# if channels[i] == 0:
#     print(mask)
# mask = mask.view(len(mask), 1).expand(-1, res.size(0)).view(len(mask), res.size(0), 1, 1).transpose(1,0)
# res += x* mask

# a = torch.ones(16,2,3,3)
# b = torch.ones(16,1,1,1)
# c = b.sum((1,2,3))
# print(c.shape)
# l = 
# y = list(range(0,2))
# print(y)
# arch_list = np.load('./save/ac_resnet20/sub_search_l1_ft1_group8_realprune/arch_iter.npy')
# acc = np.load('./save/ac_resnet20/sub_search_l1_ft1_group8_realprune/channel_layer_acc_list.npy')
# print(len(arch_list))

# # for i in range(len(arch_list)):
# #     print(i, arch_list[i])
# print(arch_list[0])
# print(arch_list[227])
# arch_linear = []
# arch_linear.append(arch_list[0])
# for i in range(len(arch_list)):
#     if i % 2 == 1 and i < 227:
#         print(i)
#         print(arch_list[i])
#         arch_linear.append(arch_list[i])
# arch_linear.append(arch_list[227])
# print(len(arch_linear))
# # np.save('./save/ac_resnet20/sub_search_pni_ft1_group4_realprune'+'/arch_iter1', arch_linear)   
# # print(len(arch_linear))
# # print(arch_linear[56])

# arch_16 = []
# for i in range(0, 114, 7):
#     print(i)
#     print(arch_linear[i])
#     arch_16.append(arch_linear[i])
# arch_16.append(arch_linear[-1])
# print(arch_16[17])
# print(len(arch_16))
# # # print(np.arange(0,170,10))
# np.save('./save/ac_resnet20/sub_search_l1_ft1_group8_realprune'+'/arch_iter_16', arch_16)   

# print(len(acc))
# acc_list = []
# arch_list1 = []
# for i in range(160):
#     temp = arch_list[i]
#     for j in range(17):
#         temp[2+j*3] = temp[2+j*3-1]
#     print(temp)
#     arch_list1.append(temp)
# print(arch_list[159])
# np.save('./save/mobilenetv2/sub_search_l1_ft50_group4_larger'+'/arch_iter1', arch_list1)   
# print(np.arange(0,170,10))
# for i in range(159):
#     if acc[i][2] > acc[i+1][2]:
#         acc_list.append(acc[i][2])
# print(acc_list)
# print(len(acc_list))
    # a = [1,2,3]
    # b = [2,3,4]
    # print(a+b)
    # print(random.randint(1,10))
# print(np.arange(0, 1.1, 0.1))
# for i in range(1,10):
#     print(i)
# print(range(1,10))
# arch_list = np.load('./save/subnet_search/resnet20/3_pni_d_ft80_group4/arch_iter.npy') #17
# print(len(arch_list))
# print(arch_list[-1])
# a = torch.randn(5)
# b, c = a.sort()
# print(b)