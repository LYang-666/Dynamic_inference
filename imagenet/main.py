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
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, change_model_width, CrossEntropyLossSoft
import models
import random
import numpy as np
from cal_flops import print_model_parm_flops
from models.noisy_layer import noise_Conv2d, noise_Linear, nas_noise_Linear, nas_noise_Conv2d, NasBatchNorm2d
from utils import Lighting, make_divisible
import math
from torch.utils.tensorboard import SummaryWriter

# try:
#     from tensorboardX import SummaryWriter
# except:
#     print("Not using tensorboard to log the training")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--lr_scheduler',
                    type=str,
                    default='linear_decaying',
                    choices=['linear_decaying', 'multistep', 'cosine_decaying'])
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--Indiv', dest='Indiv', action='store_true',
                    help='only save the model without external utils_')
parser.add_argument('--USNN', dest='USNN', action='store_true',
                    help='only save the model without external utils_')
parser.add_argument('--SUNN', dest='SUNN', action='store_true',
                    help='only save the model without external utils_')
parser.add_argument('--SNN', dest='SNN', action='store_true',
                    help='only save the model without external utils_')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# Random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# Width selection
# parser.add_argument('--u_width_ratio',
#                     type=float,
#                     default=1,
#                     help='Uniform width selection.')
parser.add_argument('--bn_calib',
                    dest='bn_calib',
                    action='store_true',
                    help='enable the one epoch batch-norm calibration')
# parser.add_argument('--uwidth_ratio_list',
#                     type=float,
#                     nargs='+',
#                     default=[1],
#                     help='uniform width ratio list to perform the training')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    # make only device #gpu_id visible, then
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU
# args.use_cuda = True
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
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')
    # # logger = Logger(tb_path)
    # try:  # try to use tensorboard summarywriter
    writer = SummaryWriter(tb_path+'/base_noise')
    # except:
    #     pass

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
        crop_scale = 0.08
        jitter_param = 0.4
        lighting_param = 0.1
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            # transforms.RandomResizedCrop(224),
            # transforms.ColorJitter(
            #     brightness=jitter_param, contrast=jitter_param,
            #     saturation=jitter_param),
            # Lighting(lighting_param),
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
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
        # arch_list = np.load('./save/sub_search_pni_ft50_group4_nodownsample/arch_iter.npy')    # 57
    
    arch_list = None
    arch_list = np.load('./save/subnet_search/mobilenetv1/test/arch_iter.npy') #41
    # arch_list = np.load('./save/subnet_search/resnet18/l1_group4_ft20/arch_iter.npy') #41
    # arch_list = np.load('./save/subnet_search/alexnet/l1_group8_ft0_conv_fc/arch_iter_conv_fc.npy') #41
    # arch_list = np.load('./save/subnet_search/vgg11/l1_group8_ft40/arch_iter.npy') #41

    # vgg 
    # arch_list = [[64, 128, 256, 256,  512, 512, 512, 512, 4096, 4096],
    #             [16, 32, 64, 64, 128, 128, 128, 128, 1024, 1024]]
    #alexnet
    # arch_list = [[64, 192, 384, 256, 256, 4096, 4096],
    #             [16, 49, 96, 64, 64, 1024, 1024]
    # ]
    #mobilenet
    curr_arch = []
    for i in range(len(arch_list[0])):
        curr_arch.append(int(arch_list[0][i]))
        # curr_arch.append(int(arch[i]))

    net = models.__dict__[args.arch](curr_arch)
    # print(curr_arch)
    # net = models.__dict__[args.arch]()

    # if args.Indiv:
    #     from torch.hub import load_state_dict_from_url
    #     pretrained_dict = load_state_dict_from_url('https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
    #     model_dict = net.state_dict()
    # # # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # # 2. overwrite entries in the existing state dict
    #     model_dict.update(pretrained_dict) 
    # # # 3. load the new state dict
    #     net.load_state_dict(model_dict)

        
    print_log("=> network :\n {}".format(net), log)
    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    soft_criterion = CrossEntropyLossSoft()
    # remove the weight-decay/penalty on sigma
    normal_param = [
        param for name, param in net.named_parameters() if not 'alpha_w' in name
    ]

    sigma_param = [
        param for name, param in net.named_parameters() if 'alpha_w' in name
    ]

    optim_config = [{
        'params': normal_param
    }, {
        'params': sigma_param,
        'weight_decay': 0
    }]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        model_params = []
        for params in net.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = state['decay']
            elif len(ps) == 2:
                weight_decay = state['decay']
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': state['learning_rate'], 'momentum': state['momentum'],
                    'nesterov': True}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)


        # optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, net.parameters()),
        #                             lr=state['learning_rate'],
        #                             momentum=state['momentum'],
        #                             weight_decay=state['decay'],
        #                             nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(filter(
            lambda param: param.requires_grad, net.parameters()),
                                        lr=state['learning_rate'],
                                        alpha=0.99,
                                        eps=1e-08,
                                        weight_decay=0,
                                        momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)
            # net.load_state_dict(checkpoint['state_dict'])

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    # set lr_schduler 
    lr_scheduler = get_lr_scheduler(optimizer)


    # change the model width
    # width_change_log = change_model_width(net, ratio=args.u_width_ratio)
    # print_log(width_change_log, log)

    # print_log(width_change_log, log)
    # print_log(
    #     "=> model width is change to: {}, model size: {:.4f}".format(
    #         args.u_width_ratio, args.u_width_ratio**2), log)

    if args.bn_calib:
        print_log("=> Perform batch-norm calibration.", log)
        batchnorm_calibration(model=net, train_loader=train_loader)

    # model accuracy evaluation

    # if args.evaluate:
    #     val_acc, val_los = validate(test_loader, net, criterion, log)  
    #     return
    #     # for i in range(2):
    #     #     if i == 0:
    #     #         curr_arch = []
    #     #         for j in range(len(arch_list[0])):
    #     #             curr_arch.append(int(arch_list[0][j]))
    #     #         net.apply(
    #     #         lambda m: setattr(m, 'ch_width', curr_arch))
    #     #         net.idx = 0
    #     #         net.module.update_model()
    #     #         print(net.ch_width)

    #     #     if i == 1:
    #     #         curr_arch = []
    #     #         for j in range(len(arch_list[0])):
    #     #             curr_arch.append(int(arch_list[51][j]))
    #     #         net.apply(
    #     #         lambda m: setattr(m, 'ch_width', curr_arch))
    #     #         net.idx = 3
    #     #         net.module.update_model()  
    #     #         print(net.ch_width)

    #     #     print(net)
    #     #     net.ch_width = curr_arch
    
    #     #     batchnorm_calibration(model=net, train_loader=train_loader)
    #     test(test_loader, net, criterion, log)  
    #     # val_acc, val_los = validate(test_loader, net, criterion, log)  
    #     return
    # if args.evaluate:
    #     for i in range(2):
    #         if i == 0:
    #             curr_arch = []
    #             for j in range(len(arch_list[0])):
    #                 curr_arch.append(int(arch_list[0][j]))
    #             net.apply(
    #             lambda m: setattr(m, 'ch_width', curr_arch))
    #             net.idx = 0
    #             net.module.update_model()
    #             print(net.ch_width)

    #         if i == 1:
    #             curr_arch = []
    #             for j in range(len(arch_list[0])):
    #                 curr_arch.append(int(arch_list[51][j]))
    #             net.apply(
    #             lambda m: setattr(m, 'ch_width', curr_arch))
    #             net.idx = 3
    #             net.module.update_model()  
    #             print(net.ch_width)

            # print(net)
            # net.ch_width = curr_arch
    
            # batchnorm_calibration(model=net, train_loader=train_loader)
            # val_acc, val_los = validate(test_loader, net, criterion, log)
    # if args.evaluate:
    #     for i in [0,3]:
    #         if i == 0:
    #             curr_arch = []
    #             for j in range(len(arch_list[0])):
    #                 curr_arch.append(int(arch_list[0][j]))
    #             net.apply(
    #             lambda m: setattr(m, 'ch_width', curr_arch))
    #             net.apply(
    #             lambda m: setattr(m, 'idx', 0))
                
    #             net.module.update_model()
    #         if i == 3:
    #             curr_arch = []
    #             for j in range(len(arch_list[0])):
    #                 curr_arch.append(int(arch_list[51][j]))
    #             net.apply(
    #             lambda m: setattr(m, 'ch_width', curr_arch))
    #             net.apply(lambda m: setattr(m, 'idx', 3))
    #             net.module.update_model()  
    #         print(net.ch_width)
    #         # print(net)
    #         # net.ch_width = curr_arch
    
    #         # batchnorm_calibration(model=net, train_loader=train_loader)
    #         val_acc, val_los = validate(test_loader, net, criterion, log)
    #     return
    # if args.evaluate:  #SUNN
    #     flops_list = []
    #     params_list = []
    #     acc_list = []
    #     width_list = np.arange(1.0, 0.2, -0.05)
    #     print(width_list)
    #     for i in range(len(width_list)):
    #     # for i in [1.0, 0.75,0.5,0.25]:
    #         print('==========')
    #         print(i)
            
    #         curr_arch = []
    #         base = arch_list[0][0]

    #         # for j in range(len(arch_list[0])):
    #         #     curr_arch.append(int(arch_list[57][j]*i))
    #         for j in range(len(arch_list[0])):
    #             # curr_arch.append(int(base * width_list[i])*int(arch_list[0][j]/base))
    #             curr_arch.append(make_divisible(int(arch_list[0][j] * width_list[i])))

    #             # curr_arch.append(int(base * i)*int(arch_list[0][j]/base))

    #         net.apply(
    #         lambda m: setattr(m, 'ch_width', curr_arch))
    #         net.apply(
    #             lambda m: setattr(m, 'idx', 4))
    #         net.module.update_model()     
    #         M_flops, M_params = print_model_parm_flops(net) 
    #         params_list.append(M_params)
    #         flops_list.append(M_flops)
    #         print('M_flops:{}, M_params:{}'.format(M_flops, M_params))
    #         print(net.ch_width)
    #         net.apply(bn_calibration_init)

    #         batchnorm_calibration(model=net, train_loader=train_loader)
    #         val_acc, val_los = validate(test_loader, net, criterion, log)
    #         acc_list.append(val_acc)
    #         print(acc_list)
    #         print(params_list)
    #         print(flops_list)
    #     return

    _Num = len(arch_list)
    if args.evaluate:
        if args.Indiv:
            validate(test_loader, net, criterion, log)
        else:
            if args.SUNN:
                flops_list = []
                params_list = []
                acc_list = []
                width_list = np.arange(1.0, 0.2, -0.05)
                print(width_list)
                for i in range(len(width_list)):
                # for i in [1.0, 0.75,0.5,0.25]:
                    print('==========')
                    print(i)
                    
                    curr_arch = []
                    base = arch_list[0][0]

                    # for j in range(len(arch_list[0])):
                    #     curr_arch.append(int(arch_list[57][j]*i))
                    for j in range(len(arch_list[0])):
                        # curr_arch.append(int(base * width_list[i])*int(arch_list[0][j]/base))
                        curr_arch.append(make_divisible(int(arch_list[0][j] * width_list[i])))

                        # curr_arch.append(int(base * i)*int(arch_list[0][j]/base))
                    net.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    net.apply(
                    lambda m: setattr(m, 'idx', 0))
                    if i == len(arch_list)-1:
                        net.apply(
                        lambda m: setattr(m, 'idx', 3))

                    net.module.update_model()     
                    M_flops, M_params = print_model_parm_flops(net) 
                    params_list.append(M_params)
                    flops_list.append(M_flops)
                    print('M_flops:{}, M_params:{}'.format(M_flops, M_params))
                    print(net.ch_width)
                    net.apply(bn_calibration_init)
                    batchnorm_calibration(model=net, train_loader=train_loader)

                    val_acc, val_los = validate(test_loader, net, criterion, log)
                    acc_list.append(val_acc)
                    print(acc_list)
                    print(params_list)
                    print(flops_list)
            elif args.USNN:
                params_list = []
                acc_list = []
                flops_list = []
                for i in np.arange(0,len(arch_list),3):
                # for i in np.arange(len(arch_list)):
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
                    if i == len(arch_list)-1:
                        net.apply(
                        lambda m: setattr(m, 'idx', 3))

                    net.module.update_model()
                    M_flops, M_params = print_model_parm_flops(net) 
                    
                    # print('evaluate one time:{}'.format(time.time()-t0))
                    print('M_flops:{}, M_params:{}'.format(M_flops, M_params))
                    print(net.ch_width)

                    # if int(M_flops) in np.arange(200, 229):

                    # # # net.ch_width = curr_arch
                    # if M_flops not in flops_list:

                    params_list.append(M_params)
                    flops_list.append(M_flops)   
                    net.apply(bn_calibration_init)
                    batchnorm_calibration(model=net, train_loader=train_loader)
                    val_acc,_ = validate(test_loader, net, criterion, log)

                    acc_list.append(val_acc)
                    print(acc_list)
                    print(flops_list)
                    print(params_list)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()



    for epoch in range(args.start_epoch, args.epochs):
        # current_learning_rate, current_momentum = adjust_learning_rate(
        #     optimizer, epoch, args.gammas, args.schedule)

        lr_scheduler.step()
        current_learning_rate = 0
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']

        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.
            format(time_string(), epoch, args.epochs, need_time,
                   current_learning_rate) +
            ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder.max_accuracy(False),
                100 - recorder.max_accuracy(False)), log)

        # # ============ TensorBoard logging ============#
        
        # we show the model param initialization to give a intuition when we do the fine tuning
        # try:
            # log the distribution.

        # except:
        #     pass

        # # ============ TensorBoard logging ============#

        # train for one epoch

        train_acc, train_los = train(train_loader, net, criterion, soft_criterion,optimizer,
                                     epoch, log, arch_list)

        # evaluate on validation set
        # evaluate the max and min model:
        # arch0 = [12,12,12,12,12,12,12,24,8,32,32,24,24,48,48,48,48,48,48]
        # arch1 = [8,8,8,8,8,8,8,16,32,8,8,24,16,32,32,32,32,32,32]
        # arch0 = [12,12,12,12,12,12,12,24,24,24,24,24,24,48,48,16,64,64,48]
        # arch1 = [8,8,8,8,8,8,8,16,16,16,16,16,16,32,32,64,16,16,32]
        # arch1 = [4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16]
        arch0 = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        arch1 = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024] 
        if args.SNN:
            for i in [0, 1, 2]:
                if i == 0:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch_list[0][j]))

                if i == 1:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch0[j]))
                if i == 2:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch1[j]))

                net.apply(
                lambda m: setattr(m, 'ch_width', curr_arch))
                net.apply(
                lambda m: setattr(m, 'idx', i))
                net.module.update_model()     
                print(net.ch_width)
                # batchnorm_calibration(model=net, train_loader=train_loader)
                val_acc, val_los = validate(test_loader, net, criterion, log)

        if args.SUNN:
            for i in [0,3]:
         
                if i == 0:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch_list[0][j]))
                    net.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    net.apply(
                    lambda m: setattr(m, 'idx', 0))
                  
                    net.module.update_model()
                if i == 3:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch_list[-1][j]))
                    net.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    net.apply(lambda m: setattr(m, 'idx', 3))
                    net.module.update_model()  
                print(net.ch_width)
                # print(net)
                # net.ch_width = curr_arch
        
                # batchnorm_calibration(model=net, train_loader=train_loader)
                
                val_acc, val_los = validate(test_loader, net, criterion, log)

        if args.USNN:
            for i in [0,3]:
                if i == 0:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch_list[0][j]))
                    net.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    net.apply(
                    lambda m: setattr(m, 'idx', 0))
                  
                    net.module.update_model()
                    print(net.ch_width)
                    val_acc, val_los = validate(test_loader, net, criterion, log)

                if i == 3:
                    curr_arch = []
                    for j in range(len(arch_list[0])):
                        curr_arch.append(int(arch_list[-1][j]))
                    net.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    net.apply(lambda m: setattr(m, 'idx', 3))
                    net.module.update_model()  
                    print(net.ch_width)

                    val_acc1, val_los1 = validate(test_loader, net, criterion, log)

        if args.Indiv:
            val_acc, val_los = validate(test_loader, net, criterion, log)

        is_best = val_acc > recorder.max_accuracy(istrain=False)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict()}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        if args.Indiv:
            
            # ## Log the weight and bias distribution
            for name, module in net.named_modules():
                name = name.replace('.', '/')
                class_name = str(module.__class__).split('.')[-1].split("'")[0]

                if hasattr(module, 'alpha_w'):   
                    if module.alpha_w is not None:
                        if module.pni == 'layerwise':
                            writer.add_scalar(name + '/alpha/',
                                            module.alpha_w.clone().item(), epoch + 1)
                        elif module.pni == 'channelwise':
                            writer.add_histogram(name+'/alpha/',
                                    module.alpha_w.clone().cpu().data.numpy(), epoch + 1, bins='tensorflow')

            writer.add_scalar('loss/train_loss', train_los, epoch + 1)
            writer.add_scalar('loss/test_loss', val_los, epoch + 1)
            writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
            writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
        
        writer.close()
        # Log the graidents distribution
        # for name, param in net.named_parameters():
        #     name = name.replace('.', '/')
        #     writer.add_histogram(name + '/grad',
        #                          param.grad.clone().cpu().data.numpy(), epoch + 1, bins='tensorflow')

        # ## Log the weight and bias distribution
        # for name, module in net.named_modules():
        #     name = name.replace('.', '/')
        #     class_name = str(module.__class__).split('.')[-1].split("'")[0]

        #     if "Conv2d" in class_name or "Linear" in class_name:
        #         if module.weight is not None:
        #             writer.add_histogram(name + '/weight/',
        #                                  module.weight.clone().cpu().data.numpy(), epoch + 1, bins='tensorflow')

        # try:
        #     # log the loss and accuracy
        #     writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        #     writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        #     writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        #     writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
        # except:
        #     pass

    # ============ TensorBoard logging ============#

    # log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, soft_criterion, optimizer, epoch, log, arch_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    if not args.Indiv:
        min_idx = len(arch_list)-1
        arch_max = arch_list[0]
        arch_min = arch_list[-1]
    end = time.time()
    train_data_size = len(train_loader.dataset)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            # the copy will be asynchronous with respect to the host.
            target = target.cuda()
            input = input.cuda()
        lr_schedule_per_iteration(optimizer, epoch, train_data_size, i)
        if args.USNN:
            optimizer.zero_grad()
            for idx in range(4):
                if idx == 0:
                    # curr_arch = arch_max
                    curr_arch = []
                    for j in range(len(arch_max)):
                        curr_arch.append(int(arch_max[j]))
                    # print(curr_arch)
                    #inpalce distillation
                    model.module.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.module.apply(lambda m: setattr(m, 'idx', idx))
                    model.module.update_model()
          
                    output = model(input)
                    loss = criterion(output, target)
                    soft_target = torch.nn.functional.softmax(output, dim=1).detach()
                    # if i % args.print_freq == 0:
                    # #     print(idx)
                    # #     print(curr_arch)

                    #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    #     print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))
                else:
                    if idx == 3:
                        # curr_arch = arch_min
                        curr_arch = []
                        for j in range(len(arch_min)):
                            curr_arch.append(int(arch_min[j]))
                    else:
                        arch_num = random.randint(1, min_idx)
                        curr_arch = []
                        for j in range(len(arch_list[arch_num])):
                            curr_arch.append(int(arch_list[arch_num][j]))
                    
                    # print('idx:{}'.format(idx))
                    # print(curr_arch)

                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))

                    # model.idx = idx
                    model.module.update_model()
    
                    # compute output
                    output = model(input)
                    loss = torch.mean(soft_criterion(output, soft_target))
                        # print(loss)
                    # if i % args.print_freq == 0:
                    #     print(curr_arch)
                    #     print(idx)
                    #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    #     print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))
                    # compute gradient and do SGD step
                loss.backward(retain_graph=True)

        if args.SUNN:
            optimizer.zero_grad()
            for idx in [0,1,2,3]:
                if idx == 0:
                    # curr_arch = arch_max
                    curr_arch = []
                    for j in range(len(arch_max)):
                        curr_arch.append(int(arch_max[j]))
                    #inpalce distillation
                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))
                    model.module.update_model()

                    output = model(input)
                    loss = criterion(output, target)
                    soft_target = torch.nn.functional.softmax(output, dim=1).detach()
                    # if i % args.print_freq == 0:
                    #     print(idx)
                    #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    #     print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))
                    # loss.backward(retain_graph=True)
                else:
                    if idx == 3:
                        curr_arch = []
                        for j in range(len(arch_min)):

                            curr_arch.append(int(arch_max[j] * 0.25))

                    else:        
                        ratio = random.uniform(0.25,1)
                        # curr_arch = arch_list[arch_num]
                        base = arch_max[0]
                        curr_arch = []
                        for j in range(len(arch_min)):
                            # curr_arch.append(int(base * ratio)*int(arch_max[j]/base))
                            curr_arch.append(make_divisible(int(arch_max[j] * ratio)))


                        # curr_arch.append(int(arch_list[arch_num][j]))
                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))

                    # model.idx = idx
                    model.module.update_model()
                    # compute output
                    output = model(input)

                    loss = torch.mean(soft_criterion(output, soft_target))
                    # if i % args.print_freq == 0 and idx ==1:
                 
                    #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    #     print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))
                loss.backward(retain_graph=True)

        if args.SNN:
            optimizer.zero_grad()

            for idx in [0,1,2]:
                if idx == 0:
                    # curr_arch = arch_max
                    curr_arch = []
                    for j in range(len(arch_max)):
                        curr_arch.append(int(arch_max[j]))
                    #inpalce distillation
                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))
                    model.module.update_model()

                    output = model(input)
                    loss = criterion(output, target)
                    # soft_target = torch.nn.functional.softmax(output, dim=1).detach()
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    if i % args.print_freq == 0:
                        print(idx)

                        print(model.ch_width)
                        # M_flops, M_params = print_model_parm_flops(model) 
                        # print(M_flops, M_params)
                        print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))
                    # loss.backward(retain_graph=True)
                if idx == 1:

                    # arch_num = idx
                    # curr_arch = arch_list[arch_num]
                    curr_arch = []
                    for j in range(len(arch_max)):
                        curr_arch.append(int(arch_mid[j]))
                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))
                    model.module.update_model()

                    # compute output
                    output = model(input)
                    loss = criterion(output, target)

                    # loss = torch.mean(soft_criterion(output, soft_target))
                if idx == 2:

                    # arch_num = idx
                    # curr_arch = arch_list[arch_num]
                    curr_arch = []
                    for j in range(len(arch_max)):
                        curr_arch.append(int(arch_min[j]))
                    model.apply(
                    lambda m: setattr(m, 'ch_width', curr_arch))
                    model.apply(lambda m: setattr(m, 'idx', idx))
                    model.module.update_model()

                    # compute output
                    output = model(input)
                    loss = criterion(output, target)                
                    # if i % args.print_freq == 0:
                    #     print(idx)

                    #     print(model.ch_width)
                    #     # M_flops, M_params = print_model_parm_flops(model) 
                    #     # print(M_flops, M_params)
                    #     print('idx:{}, acc:{}, loss:{}'.format(idx, prec1, loss.item()))

                loss.backward(retain_graph=True)
        if args.Indiv:
            # curr_arch = []
            # for j in range(len(arch_max)):
            #     curr_arch.append(int(arch_max[j]))
            # #inpalce distillation
            # model.apply(
            # lambda m: setattr(m, 'ch_width', curr_arch))
            # model.apply(lambda m: setattr(m, 'idx', 0))
            # model.update_model()
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()

            loss.backward()

        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                current_learning_rate = param_group['lr']
            print('current learning rate:{}'.format(current_learning_rate))
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    time_acc = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()
                # compute output

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg, losses.avg

def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        # if getattr(FLAGS, 'cumulative_bn_stats', False):
        # m.momentum = None
def bn_calibration_false(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = False

def test(val_loader, model, criterion, log):


    # switch to evaluate mode
    model.eval()
    time_acc = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i < 100:
                if args.use_cuda:
                    target = target.cuda()
                    input = input.cuda()
                    # compute output
                t0 = time.time()
                output = model(input)
                if i > 0:
                    # t0 = time.time()
                    # model(input)
                    time_acc += time.time() -t0
                    print(time.time() -t0)
            else:
                break
    # print(len(val_loader))
    print('acc_time:{}'.format(time_acc/99))
    # return top1.avg, losses.avg


def batchnorm_calibration(model, train_loader):
    '''feed the model with train dataloader in train model (update running_mean/var)'''
    model.train()  # in train mode, to calbirate the BN
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(train_loader):
            if batch_idx == 200:
                break
            if args.use_cuda:
                input = input.cuda()

            output = model(input)

    # finish the calib, convert back to eval mode
    model.eval()


def get_lr_scheduler(optimizer):
    """get learning rate"""
    warmup_epochs = 0
    print(args.lr_scheduler, args.epochs)
    if args.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90],
            gamma=0.1)
    elif args.lr_scheduler == 'linear_decaying':
        num_epochs = args.epochs - warmup_epochs
        lr_dict = {}
        for i in range(args.epochs):
            lr_dict[i] = 1. - (i - warmup_epochs) / num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif args.lr_scheduler == 'cosine_decaying':
        num_epochs = args.epochs - warmup_epochs
        lr_dict = {}
        for i in range(args.epochs):
            lr_dict[i] = (
                1. + math.cos(
                    math.pi * (i - warmup_epochs) / num_epochs)) / 2.
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(args.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    args.lr_scheduler))
    return lr_scheduler


def lr_schedule_per_iteration(optimizer, epoch, train_data_size, batch_idx=0):
    """ function for learning rate scheuling per iteration """
    warmup_epochs = 0
    num_epochs = args.epochs - warmup_epochs
    iters_per_epoch = train_data_size / args.batch_size
    current_iter = epoch * iters_per_epoch + batch_idx + 1
    # if getattr(FLAGS, 'lr_warmup', False) and epoch < warmup_epochs:
    #     linear_decaying_per_step = FLAGS.lr/warmup_epochs/iters_per_epoch
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = current_iter * linear_decaying_per_step
    if args.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = args.learning_rate/num_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] -= linear_decaying_per_step
    elif args.lr_scheduler == 'cosine_decaying':
        mult = (
            1. + math.cos(
                math.pi * (current_iter - warmup_epochs * iters_per_epoch)
                / num_epochs / iters_per_epoch)) / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * mult
    else:
        pass


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
