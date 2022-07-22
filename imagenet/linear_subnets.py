import torch
import numpy as np
import random

arch_list = np.load('./save/resnet20/sub_search_l1_ft50_group4_larger/arch_iter.npy')
acc = np.load('./save/resnet20/sub_search_l1_ft50_group4_larger/channel_layer_acc_list.npy')

# print(arch_list[57])
# # print(acc)
# arch_index = []
# for i in range(0,58):
#     acc_crr = acc[i][2]
#     acc_next = acc[i+1][2]
#     if acc_crr < acc_next and (acc_crr+10) > acc_next:
#         continue
#     else:
#         arch_index.append(i)
# print(arch_index)

# arch_inter_linear = [[16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0]]
# for i in arch_index:
#     arch_inter_linear.append(arch_list[i])

# print(len(arch_inter_linear))
# print(arch_inter_linear[0])

# print(arch_inter_linear[52])

# np.save('./save/resnet20/sub_search_l1_ft50_group4_larger'+'/arch_iter_linear', arch_inter_linear)   
arch_index = [0]
# ./save/resnet20/sub_search_pni_ft50_group4_larger_4channels
arch_list1 = np.load('./save/resnet20/sub_search_pni_ft50_group1_larger/arch_iter.npy')
acc = np.load('./save/resnet20/sub_search_pni_ft50_group1_larger/channel_layer_acc_list.npy')

# print(len(arch_list1))
print(arch_list1[0])
print(arch_list1[516])

for i in range(1,517):
    acc_crr = acc[i][2]
    acc_next = acc[i+1][2]
    if acc_crr < acc_next and (acc_crr+10) > acc_next:
        continue
    else:
        arch_index.append(i)
print(len(arch_index))
arch_inter_linear = []
for i in arch_index:
    arch_inter_linear.append(arch_list1[i])
print(arch_inter_linear[0])
print(arch_inter_linear[327])
# print(arch_index)
np.save('./save/resnet20/sub_search_pni_ft50_group1_larger'+'/arch_iter_linear', arch_inter_linear)   

# arch_list1 = np.load('./save/resnet20/sub_search_pni_ft50_group4_larger_testdata/arch_iter.npy')
# for i in np.arange(0,515,32):
#     arch_index.append(i)
# arch_index.append(516)

# arch_inter_linear = []
# for i in arch_index:
#     arch_inter_linear.append(arch_list1[i])
#     print(arch_list1[i])
# print(len(arch_inter_linear))
# print(arch_inter_linear[0])
# print(arch_inter_linear[17])
# np.save('./save/resnet20/sub_search_pni_ft50_group1_larger'+'/arch_iter_linear_17', arch_inter_linear)   
