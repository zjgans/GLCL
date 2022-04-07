from __future__ import print_function

import argparse
import socket
import time
import os
import mkl


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate, embedding
from dataloader import get_dataloaders

mkl.set_num_threads(2)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default="", help='absolute path to .pth model')
    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR-FS', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "toy",'cub'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # specify data_root
    parser.add_argument('--data_root', type=str, default='/home/~/dataset', help='path to data root')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    
    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    
    opt = parser.parse_args()
    
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    if opt.dataset == 'cub':
        opt.transform = 'E'
        
    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        if(opt.dataset=="toy"):
            opt.data_root = '{}/{}'.format(opt.data_root, "CIFAR-FS")
        else:   
            opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    return opt

def main():

    opt = parse_option()

    opt.n_test_runs = 600
    val_loader, meta_testloader, meta_valloader, n_cls, no_sample,dataset = get_dataloaders(opt)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset,n_trans=4,embd_sz=128)
    ckpt = torch.load(opt.model_path)["model"]

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.","")
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)

    # model.load_state_dict(ckpt["model"])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, is_feat=True)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

if __name__ == '__main__':
    main()
