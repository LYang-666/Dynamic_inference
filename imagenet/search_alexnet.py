from __future__ import division
from __future__ import absolute_import

import os
import sys
import shutil
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from cal_flops import print_model_parm_flops
from collections import OrderedDict
from torch.utils.data import random_split

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import models
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from logger import Logger
from models.layers import aw_DownsampleA

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', default='/home/elliot/data/pytorch/svhn/',
                    type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='lbcnn', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--optimizer', type=str, default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--model_only', dest='model_only', action='store_true',
                    help='only save the model without external utils_')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=5000, help='manual seed')

parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')

parser.add_argument('--acc', dest='acc', action='store_true',
                    help='only save the model without external utils_')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# if args.ngpu == 1:
#     # make only device #gpu_id visible, then
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True


###############################################################################
###############################################################################

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path,
                            'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')
    logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.MNIST(args.data_path, train=False,
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train',
                               transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test',
                              transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(
            args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test',
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)


    # split validate data
    length = [10000, len(test_data)-10000]
    length1 = [int(len(train_data)/4), len(train_data)-int(len(train_data)/4)]
    vali_data, tes_data = random_split(test_data, length)
    train_data1, _ =  random_split(train_data, length1)
    train_loader = torch.utils.data.DataLoader(train_data1, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch]()
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net)

    # define loss function (criterion) and optimizer

    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        # optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, net.parameters()),
        #                             lr=state['learning_rate'],
        #                             momentum=state['momentum'], weight_decay=0.0, nesterov=True)
        optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, net.parameters()),
                                    lr=state['learning_rate'],
                                    momentum=0, weight_decay=0, nesterov=False)
    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "YF":
        print("using YellowFin as optimizer")
        optimizer = YFOptimizer(filter(lambda param: param.requires_grad, net.parameters()), lr=state['learning_rate'],
                                mu=state['momentum'], weight_decay=state['decay'])

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(filter(lambda param: param.requires_grad, net.parameters()),
                                        lr=state['learning_rate'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
 
    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches
    # for name, value in net.named_parameters():
    #     print(name)
    # optionally resume from a checkpoint
    if args.resume:
        new_state_dict = OrderedDict()
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                # optimizer.load_state_dict(checkpoint['optimizer'])
            state_tmp = net.state_dict()
            for k, v in checkpoint['state_dict'].items():
                name = k
                print(name)
                new_state_dict[name] = v
            if 'state_dict' in checkpoint.keys():
                #state_tmp.update(new_state_dict['state_dict'])
                state_tmp.update(new_state_dict)
            else:
                print('loading from pth file not tar file')
                state_tmp.update(new_state_dict)
                #state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)

            print_log("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    
    # acc_base, _ = validate(test_loader, net, criterion, log)
    # print_log('base accuracy:{}'.format(acc_base), log)

    # start NAS
    
    # Main loop
    start_time = time.time()
    
    # for n, v in net.items():

    # calculate the number of iteration
    num_iter = 0
    num = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            print(name)
            num+=1
            num_iter += m.out_channels
        if isinstance(m, nn.Linear):
            if m.out_features != 1000:
                print(name)
                num+=1   
                num_iter += m.out_features

    print_log('Number of iteration:{}'.format(num_iter), log)
    
    l = 0
    layer = 0
    channel = 0
    channel_layer_acc_list = []
    arch_iter = []
    index_list = torch.zeros(num)
    index_list_inverse = torch.zeros(num)

    m_channels = torch.zeros(num)
    ll = 0
    l_skip = torch.zeros(num)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size[0] is not 1:

                m_channels[ll] = int(m.out_channels)
                ll += 1
        if isinstance(m, nn.Linear):
            if m.out_features != 1000:
                print(m.out_features)

                m_channels[ll] = int(m.out_features)
                ll += 1         

    print(m_channels)

    t0 = time.time()
    l_c_list = []

    for m in range(num):
        one_list = []
        l_c_list.append(one_list)
            
    t0 = time.time()
    iter_time = AverageMeter()
    
    # main searching loop 
    group = 8
    _Num_iter = num * group
    arch_iter.append(m_channels.tolist())

    for num in range(_Num_iter):
        
        need_hour, need_mins, need_secs = convert_secs2time(
            iter_time.avg * (_Num_iter - num))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        t1 = time.time()
        l = 0
        acc1 = 0.0
        loss1 = 100000
        t_valid = 0
        print('=========  iteration :{} =========='.format(num))
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                if l_skip[l] == 1:
                    l += 1
                    continue
                
                # print('Layer: {} {}'.format(l, m))
                # l1 norm based
                w, index = m.weight.data.abs().sum(dim=(1,2,3)).sort()
    
                init_idx = int(index_list[l])
                current_idx = list(range(init_idx, init_idx+int(m_channels[l]/group)))                    
    
                # pni based 

                # alpha, index = m.alpha_w.data.squeeze().abs().sort()
                # init_idx = int(index_list[l])
                # current_idx = list(range(init_idx, init_idx+int(m_channels[l]/group)))
                # if l == 2:
                #     x = m.alpha_w.data.squeeze().abs()
                #     min_v = x.min()
                #     range_v = x.max() - x.min()
                #     x = (x - min_v)/range_v
                
                #     print('x:{}'.format(x)) 

                # x = x/x.sum()
                # alpha, index = m.alpha_w.data.squeeze().abs().sort(descending=True)
                # current_idx = list(range(int(m_channels[l]/group)))

                # #update the num_channels in current layer
    
                # # number of channels reduces 1                    
                # # the pruned channel 
                net.module.ch_width[l] -= int(m_channels[l]/group)
                net.module.ch_index[l].extend(index[current_idx].tolist())
                net.module.update_model(l)
                # t2 = time.time()
                # updateBN(vali_loader, net)
                acc, loss = validate(vali_loader, net, criterion, log)
                
                print('Layer: {},  Num_Channel: {},  Acc: {}'.format(l, net.module.ch_width[l],  acc))
                # print(index[current_idx].tolist())
                # print('l1_'+ str(l)+'= {}'.format(m.weight.data.abs().sum(dim=(1,2,3))))
                # print('mean_'+ str(l)+'= {}'.format(m.weight.data.abs().mean())))
                # print('var_'+ str(l)+'= {}'.format(m.weight.data.abs().var().numpy())))

                # t_valid += (time.time() - t2)

                # recover 
                net.module.ch_width[l] += int(m_channels[l]/group) 
                # net.ch_index[l] = [-1]

                net.module.ch_index[l] = [i for i in net.module.ch_index[l] if i not in index[current_idx].tolist()]
                # net.ch_index[l].remove(i for i in [index[current_idx].tolist()]) 
                # print(net.ch_index)

                net.module.update_model(l)

                if args.acc:
                    if acc > acc1:
                        acc1 = acc
                        layer = l
                        channel = index[current_idx].tolist()

                    l += 1
                else:
                    if loss < loss1:
                        loss1 = loss
                        layer = l
                        channel = index[current_idx].tolist()
                    l += 1
            
             #after prune conv layer than prune fc layer
            if l_skip[:5].sum() == len(l_skip[:5]):
                if isinstance(m, nn.Linear):
                    if l_skip[l] == 1:
                        l += 1
                        continue
                    
                    # print('Layer: {} {}'.format(l, m))
                    # l1 norm based
        
                    w, index = m.weight.data.abs().sum(dim=(1)).sort()
            
                    init_idx = int(index_list[l])
                    current_idx = list(range(init_idx, init_idx+int(m_channels[l]/group)))                    
        
                    # pni based 

                    # alpha, index = m.alpha_w.data.squeeze().abs().sort()
                    # init_idx = int(index_list[l])
                    # current_idx = list(range(init_idx, init_idx+int(m_channels[l]/group)))
                    # if l == 2:
                    #     x = m.alpha_w.data.squeeze().abs()
                    #     min_v = x.min()
                    #     range_v = x.max() - x.min()
                    #     x = (x - min_v)/range_v
                    
                    #     print('x:{}'.format(x)) 

                    # x = x/x.sum()
                    # alpha, index = m.alpha_w.data.squeeze().abs().sort(descending=True)
                    # current_idx = list(range(int(m_channels[l]/group)))

                    # #update the num_channels in current layer
        
                    # # number of channels reduces 1                    
                    # # the pruned channel 
                    net.module.ch_width[l] -= int(m_channels[l]/group)
                    net.module.ch_index[l].extend(index[current_idx].tolist())
                    net.module.update_model(l)
                    # t2 = time.time()
                    # updateBN(vali_loader, net)
                    acc, loss = validate(vali_loader, net, criterion, log)
                    
                    print('Layer: {},  Num_Channel: {},  Acc: {}'.format(l, net.module.ch_width[l],  acc))
                    # print(index[current_idx].tolist())
                    # print('l1_'+ str(l)+'= {}'.format(m.weight.data.abs().sum(dim=(1,2,3))))
                    # print('mean_'+ str(l)+'= {}'.format(m.weight.data.abs().mean())))
                    # print('var_'+ str(l)+'= {}'.format(m.weight.data.abs().var().numpy())))

                    # t_valid += (time.time() - t2)

                    # recover 
                    net.module.ch_width[l] += int(m_channels[l]/group) 
                    # net.ch_index[l] = [-1]

                    net.module.ch_index[l] = [i for i in net.module.ch_index[l] if i not in index[current_idx].tolist()]
                    # net.ch_index[l].remove(i for i in [index[current_idx].tolist()]) 
                    # print(net.ch_index)

                    net.module.update_model(l)

                    if args.acc:
                        if acc > acc1:
                            acc1 = acc
                            layer = l
                            channel = index[current_idx].tolist()

                        l += 1
                    else:
                        if loss < loss1:
                            loss1 = loss
                            layer = l
                            channel = index[current_idx].tolist()
                        l += 1               
        # print("validate time:{}".format(t_valid))

        #if in current iteration, 
        if index_list[layer] == m_channels[layer]-int(m_channels[layer]/group)*3:
            l_skip[layer] = 1
            index_list[layer] += int(m_channels[layer]/group)
        else:
     
            index_list[layer] += int(m_channels[layer]/group)
        
        # prune the net 
        net.module.ch_width[layer] -= int(m_channels[layer]/group) 

        for i in channel:
            if i in net.module.ch_index[layer]:
                print('current channel index alrady in net.ch_index')
                break
        net.module.ch_index[layer].extend(channel)
        # net.ch_index[layer] = channel
        net.module.update_model(layer)

        # calculate params, flops
        # M_flops, M_params = print_model_parm_flops(net) 
        # print_log("M_flops:{}, M_params:{}".format(M_flops, M_params), log)
        # if num == 0:
        #     max_flop = M_flops
        # smilate time

        print_log("index_list:{}".format(index_list), log)
        print("layer_skip:{}".format(l_skip))

        # inverse index_list
        for i in range(len(index_list)):
            index_list_inverse[i] = m_channels[i]-index_list[i]
        arch_iter.append(index_list_inverse.tolist())

        print_log("index_list_inverse:{}".format(index_list_inverse.tolist()), log)

        channel_layer_acc_list.append([layer, len(channel), acc])

        #l_c_list array list all pruned channles index in current layers
        # print(layer)
        # print(channel)
        l_c_list[layer].extend(channel)
        # print("l_c_list:{}".format(l_c_list))


        # prune_channel(net, l_c_list, layer)
        # for name, m in net.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         print(name, m.weight.data.size(), m.out_channels, m.stride)

        print_log('run one iteration: {} seconds'.format(time.time() - t1), log)
        nas_logger(base_dir=args.save_path,
                   iter=i,
                   layer=layer,
                   channel=channel,
                   accuracy=acc1)
        
        if acc1 < 20:
            fine_tuning(train_loader, vali_loader, net, l_c_list, criterion,optimizer, writer, recorder, log, i)
        #update BN
        # else:
        #     print('start update Batchnorm')
        #     updateBN(vali_loader, net)

        print_log(
            '\n==>>{:s} [Iteration={:03d}/{:03d}] {:s}'.
            format(time_string(), num, _Num_iter, need_time), log)
        print_log(" [Current iteration:{}] | [The layer:{}] | [The channel:{}] | [Current accuracy:{:2f}] ".
            format(num, layer, channel, acc1), log)

        iter_time.update(time.time() - start_time)
        start_time = time.time()

        print_log('run one iteration: {} seconds'.format(time.time() - t1), log)
        print_log('accumulate iteration: {} seconds'.format(time.time() - t0), log)

        np.save(args.save_path+'/channel_layer_acc_list', channel_layer_acc_list)
        np.save(args.save_path+'/arch_iter', arch_iter)    

        if l_skip.sum() == len(l_skip):
            break
    print('{} seconds'.format(time.time() - t0))
    # print_log(channel_layer_acc_list, log)
    channel_layer_acc_list = np.array(channel_layer_acc_list)
    # np.save('channel_layer_acc_list', channel_layer_acc_list)
    # np.save(args.save_path+'/channel_layer_acc_list', channel_layer_acc_list)

    # np.save(args.save_path+'/arch_iter', arch_iter)

    # ============ TensorBoard logging ============#

    log.close()

def do_comparison(net, index_list, l_skip, m_channels, criterion):
    
    l = 0
    acc1 = 0.0
    loss1 = 100000
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size[0] is not 1:
                if l_skip[l] == 1:
                    l += 1
                    continue
                
                # print('Layer: {} {}'.format(l, m))
                alpha, index = m.alpha_w.data.squeeze().abs().sort(descending=True)
                current_idx = 0
                #update the num_channels in current layer
      
                # number of channels reduces 1
                net.ch_width[l] -= 1
                # the pruned channel index
                net.ch_index[l] = [index[current_idx].tolist()]
                acc, loss = validate(vali_loader, net, criterion, log)
                # recover 
                net.ch_width[l] += 1
                net.ch_index[l] = [-1]
                # m.weight.data[index[current_idx]] = tmp_w
                if args.acc:
                    if acc > acc1:
                        acc1 = acc
                        layer = l
                        channel = int(index[current_idx])

                    l += 1
                else:
                    if loss < loss1:
                        loss1 = loss
                        layer = l
                        channel = int(index[current_idx])
                    l += 1
    #if in current iteration, 
    if index_list[layer] == m_channels[layer]-2:
        l_skip[layer] = 1
        index_list[layer] += 1
    else:
    
        index_list[layer] += 1

    return index_list, layer, channel, acc1

def prune_channel(net, l_c_list, layer):
        # index conv layer
        l = 0  # conv
    
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                if l == layer:
                # if len(l_c_list[l]) is not 0:

                    channel_width = l_c_list[l]
                    # # number of channels reduces 1
                    # net.ch_width[l] = m.max_out_ch - len(channel_width)
                    # # the pruned channel index
                    # net.ch_index[l] = channel_width     
                    # net.update_model(layer)         
                    m.alpha_w.data[channel_width] = 0
                    # m.weight.data[channel_width] = 0
                    # prune the output channel of current layer
                l += 1


# def prune_channel(net, l_c_list):
#         # index conv layer
#         l = 0  # conv
#         b = 0 # bn
#         d = 0
    
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.kernel_size[0] is not 1:
#                     if l > 0:
#                         if len(l_c_list[l-1]) is not 0:
#                             channel_width = l_c_list[l-1]
#                             # prune the input channel of current layer
#                             m._update_n_channels(channel_width, in_ch=True)
#                     if len(l_c_list[l]) is not 0:

#                         channel_width = l_c_list[l]
#                         m.alpha_w.data[channel_width] = 0
#                         m.weight.data[channel_width] = 0
#                         # prune the output channel of current layer
#                         m._update_n_channels(channel_width, out_ch=True) 
#                     l += 1
#             # if isinstance(m, aw_DownsampleA):
#             #     if len(l_c_list[l-1]) is not 0:
#             #         channel_width = l_c_list[l-1]
#             #         m._update_n_channels(channel_width) 
#             if isinstance(m, nn.Linear):
#                 if len(l_c_list[18]) is not 0:            
#                     channel_width = l_c_list[18]
#                     m._update_n_channels(channel_width)

#             if isinstance(m, nn.BatchNorm2d):
#                 if len(l_c_list[b]) is not 0:
#                     channel_width = l_c_list[b]

#                     # prune the output channel of current layer
#                     m._update_n_channels(channel_width)    
#                 b += 1                


def countZeroChannles(model):
    zero = 0
    l = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size[0] is not 1:

                for c in range(m.out_channels):
                    # print(m.out_channels)
                    if m.weight.data[c].sum() == 0.:
                        zero +=1
    return zero

def countZeroWeights(model):
    zeros = 0
    total = 0
    for param in model.parameters():
        if len(param.shape) == 4 and param.shape[1] != 1:
            if param is not None:
                # print(param.shape)
                zeros += torch.sum((param == 0.).int()).item()
                total += param.numel()
    print("Current zeros:{}".format(zeros))
    print("Current total:{}".format(total))

    return (zeros/total)*1.0

def updateBN(train_loader, model):

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        model(input_var)

        # # measure elapsed time
        # end = time.time()

def fine_tuning(train_loader, vali_loader, net,  l_c_list, criterion,optimizer, writer, recorder, log, i):

        epoch_time = AverageMeter()

        # print('net skip:{}'.format(net.skip))
        run_epochs = 0
        # if i > 0 and i % 50 == 0:
        print_log("start fine tuning", log)
        for epoch in range(args.epochs):
            current_learning_rate, current_momentum = adjust_learning_rate(
                optimizer, epoch, args.gammas, args.schedule)
            # Display simulation time
            need_hour, need_mins, need_secs = convert_secs2time(
                epoch_time.avg * (args.epochs - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs)

            print_log(
                '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                    need_time, current_learning_rate,
                                                                                    current_momentum)
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                100 - recorder.max_accuracy(False)), log)  

            val_acc, val_los = validate(vali_loader, net, criterion, log)            
            print('test_accuracy:{}, test_los:{}'.format(val_acc, val_los))

            # train for one epoch          
            train_acc, train_los = train(
                train_loader, net, criterion, optimizer, epoch, log, l_c_list)
            print('train_acc:{}, train_los:{}'.format(train_acc, train_los))

    
            # evaluate on validation set
            val_acc, val_los = validate(vali_loader, net, criterion, log)            
            print('test_accuracy:{}, test_los:{}'.format(val_acc, val_los))

            # ## Log the weight and bias distribution
            for name, module in net.named_modules():
                name = name.replace('.', '/')
                class_name = str(module.__class__).split('.')[-1].split("'")[0]

                if hasattr(module, 'alpha_w'):   
                    if module.alpha_w is not None:
                        if module.pni is 'layerwise':
                            writer.add_scalar(name + '/alpha/',
                                            module.alpha_w.clone().item(), i + 1)
                        elif module.pni is 'channelwise':
                            writer.add_histogram(name+'/alpha/',
                                    module.alpha_w.clone().cpu().data.numpy(), i + 1, bins='tensorflow')


            writer.add_scalar('loss/train_loss', train_los, i + 1)
            writer.add_scalar('loss/test_loss', val_los, i + 1)
            writer.add_scalar('accuracy/train_accuracy', train_acc, i + 1)
            writer.add_scalar('accuracy/test_accuracy', val_acc, i + 1)
            if val_acc > 20.0:
                run_epochs = epoch
                break
            if epoch == 0:
                break
        print_log("Number of training epochs:{}".format(run_epochs), log)


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            output = model(input)
            # for m in model.modules():
            #     if isinstance(m, nn.Conv2d):
            #         print(len(m.sel_out_channels))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    return top1.avg, losses.avg



# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, layer_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    print('train mode')


    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
   
        data_time.update(time.time() - end)
        if args.use_cuda:
            # the copy will be asynchronous with respect to the host.
            target = target.cuda(async=True)
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        l = 0
        for name, m in model.named_modules():
            # print(name)
            if isinstance(m, nn.Conv2d):
                if len(layer_list[l]) is not 0:
                    channel_width = layer_list[l]
                    # m.weight.data[channel_width] = 0         
                    # m.weight.grad.data[channel_width] = 0
                    m.alpha_w.data[channel_width] = 0
                    m.alpha_w.grad.data[channel_width] = 0
                l += 1
       
            if isinstance(m, nn.Linear):

                if l is not 7:
                    if len(layer_list[l]) is not 0:
                        channel_width = layer_list[l]
                        # m.weight.data[channel_width] = 0         
                        # m.weight.grad.data[channel_width] = 0
                        m.alpha_w.data[channel_width] = 0
                        m.alpha_w.grad.data[channel_width] = 0
                l+=1
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                    'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def nas_logger(base_dir, iter, layer, channel, accuracy):
    file_name = 'nas.txt'
    file_path = "%s/%s" % (base_dir, file_name)   
     # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('iteration layer channel accuracy\n')
        create_log.close()

    recorder = {}
    recorder['iteration'] = iter
    recorder['layer'] = layer
    recorder['channel'] = channel
    recorder['accuracy'] = accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{iteration}     {layer}    {channel}     {accuracy}\n'.format(**recorder))   




if __name__ == '__main__':
    main()





