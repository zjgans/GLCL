from __future__ import print_function

import argparse
import socket
import time
import os
import mkl
import seaborn as sns
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet

from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list
from local_branch.aggregation import Aggregation
mkl.set_num_threads(2)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str,
                        default="./save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_gamma_0.6_eta_0.8_90_1/ckpt_epoch_90.pth",
                        help='absolute path to .pth model')
    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                         'CIFAR-FS', 'FC100', "toy"])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # specify data_root
    parser.add_argument('--data_root', type=str, default='/home/lxj/new_main/dataset', help='path to data root')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--type',type=str,default='baseline',help='the name of ablation test')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=1, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=1, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=1, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        if (opt.dataset == "toy"):
            opt.data_root = '{}/{}'.format(opt.data_root, "CIFAR-FS")
        else:
            opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    return opt

def get_tsne_dataset(opt):

    if opt.dataset =='CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_test_options[opt.transform]

        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.dataset == 'CIFAR-FS':
            n_cls = 64
        elif opt.dataset == 'FC100':
            n_cls = 60
    else:
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        n_cls = 64

    return meta_testloader,meta_valloader,n_cls

def main():
    opt = parse_option()
    meta_test_loader,meta_val_loader,n_cls= get_tsne_dataset(opt)

    model = create_model(opt.model, n_cls, opt.dataset,n_trans=4)
    ckpt = torch.load(opt.model_path)["model"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    multibranch_model = Aggregation(model).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    all_features,all_ys = get_feature_and_label(multibranch_model,iter(meta_test_loader),is_feat=True,is_norm=True)

    save_path = 'heatmap_intra.jpg'
    # sns_plot(all_features)
    sns_plot_sim(all_features,all_features,save_path)
    # sns_global_local(all_features,all_features)

def sns_plot(features):
    save_path = 'heatmap_intra.jpg'
    print('features',features.shape)
    df = pd.DataFrame(features[:5].reshape(640,5))
    convariance_matrix = df.corr()
    fig = sns.heatmap(convariance_matrix,cmap='Red',annot=False)
    heatmap = fig.get_figure()
    heatmap.savefig(save_path,dpi=400)

def sns_plot_sim(g_feat,l_feat,save_path):

    g_feat = torch.Tensor(g_feat[:5])
    l_feat = torch.Tensor(l_feat[:5])
    sim_matrix = torch.einsum('nk,mk->nm',g_feat,l_feat)
    print('sim_matrix',sim_matrix.size())
    sim_matrix = sim_matrix.numpy()
    print('sim_matrix',sim_matrix.shape)
    # sns.set(font_scale=1.5)
    # sns.color_palette("YlOrRd", as_cmap=True)
    fig = sns.heatmap(sim_matrix,cmap='Blues', annot=True)   #OrRd_r
    heatmap = fig.get_figure()
    heatmap.savefig(save_path, dpi=400)

def get_feature_and_label(net,testloader,is_feat,is_norm):
    net= net.eval()
    all_features,all_ys = [],[]
    all_local_features = []
    data = next(testloader)

    with torch.no_grad():

            support_xs,support_ys,query_xs,query_ys = data

            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()

            batch_size,_,height,width,channel = support_xs.size()
            support_xs = support_xs.view(-1,height,width,channel)
            query_xs = query_xs.view(-1,height,width,channel)

            if is_feat:
                feat_support, _ = net.encoder(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net.encoder(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)
            else:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)


            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)


            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()


            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            features = np.concatenate([support_features,query_features])
            labels = np.concatenate([support_ys,query_ys])
            all_features.append(features)

            all_ys.append(labels)

    all_features = np.concatenate(all_features)
    all_ys = np.concatenate(all_ys)

    return all_features,all_ys


def tsne_plot(all_features,all_ys,title=""):
    all_transformed = TSNE(n_jobs=20,metric='cosine',square_distances=True).fit_transform(all_features)
    num_features = len(all_features)
    plt.figure()
    plt.xticks()
    plt.yticks()

    plt.scatter(all_transformed[:num_features,0],all_transformed[:num_features,1],c=all_ys,cmap="Accent",s=5)  #cmap="tab10"

    # plt.title(title,y=-2)
    plt.savefig(f"{title}.jpg")
    plt.show()

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

if __name__ == '__main__':
    main()


