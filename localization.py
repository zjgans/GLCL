from __future__ import print_function

import argparse
import socket
import time
import os
import mkl
from PIL import  Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet,ImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100,CIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

mkl.set_num_threads(2)
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str,
                        default="./save/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_compare_T_0.2_0.99/ckpt_epoch_65.pth",
                        help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                         'CIFAR-FS', 'FC100', "toy"])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # specify data_root
    parser.add_argument('--data_root', type=str, default='/data/zhoujun/project/dataset/data', help='path to data root')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=5, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=1, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')

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

class Preprocessor(Dataset):
    def __init__(self, dataset, transforms=None):
        super(Preprocessor, self).__init__()
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label, _ = self.dataset[item]

        img = self.transforms(img)

        return img, label

def get_heatmap_dataset(opt):
    if opt.dataset == "miniImageNet":
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ToTensor(),
            normalize
        ])

        dataset = ImageNet(args=opt, partition='train', transform=transform)
        data_loader = DataLoader(Preprocessor(dataset=dataset.img_label,transforms=transform),
                                 batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
                                 num_workers=opt.num_workers
                                 )
        n_cls = 64

    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize= transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize
        ])
        dataset = CIFAR100(args=opt, partition='train', transform=transform)
        data_loader = DataLoader(Preprocessor(dataset=dataset.img_label, transforms=transform),
                                 batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
                                 num_workers=opt.num_workers
                                 )
        n_cls = 64

    return data_loader,n_cls

def main():
    opt = parse_option()

    train_loader, n_cls = get_heatmap_dataset(opt)

    model = create_model(opt.model, n_cls, opt.dataset,n_trans=4)
    ckpt = torch.load(opt.model_path)["model"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    activate_fun = nn.Softmax(dim=1)
    if torch.cuda.is_available():
        model = model.cuda()
        activate_fun = activate_fun.cuda()
        cudnn.benchmark = True

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        def attention_forward(encoder, imgs):
            # hard-coded forward because we need the feature-map and not the finalized feature
            x = encoder.layer1(imgs)
            x = encoder.layer2(x)
            x = encoder.layer3(x)
            feats = encoder.layer4(x)
            # feats = encoder.avgpool2(x)
            feats_as_batch = feats.permute((0, 2, 3, 1)).contiguous().view((-1, feats.shape[1]))
            # reminder: "fc" layer outputs: (feature, class logits)
            # feats_as_batch = encoder.classifier(feats_as_batch)
            feats_as_batch = activate_fun(feats_as_batch)
            # feats_as_batch = F.normalize(feats_as_batch,dim=0)

            feats_as_batch = feats_as_batch.view(
                (feats.shape[0], feats.shape[2], feats.shape[3], feats_as_batch.shape[1]))
            feats_as_batch = feats_as_batch.permute((0, 3, 1, 2))
            print(feats_as_batch.size())
            return feats_as_batch

        f_q = attention_forward(model, images)
        localization(images, f_q, opt.batch_size, batch_id=i, img_size=84)
        if i == 100:
            break
def localization(im_q, f_q, batch_size, batch_id, img_size):
    os.makedirs('imgs', exist_ok=True)
    for idd in range(batch_size):
        aa = torch.norm(f_q, dim=1)

        imgg = im_q[idd] * torch.Tensor([[[0.229, 0.224, 0.225]]]).view(
            (1, 3, 1, 1)).cuda() + torch.Tensor(
            [[[0.485, 0.456, 0.406]]]).view((1, 3, 1, 1)).cuda()
        heatmap = F.interpolate(((aa[idd]-aa[idd].min() )/ (aa[idd].max()-aa[idd].min())).detach().unsqueeze(0).unsqueeze(0).repeat((1, 3, 1, 1)),
                                [img_size, img_size],mode='bilinear')

        thresh = 0.3
        heatmap[heatmap < thresh] = 0

        plt.imsave(f'imgs/bImg_{idd}_batch_100_{batch_id}.png',
                   torch.cat((imgg, heatmap * imgg), dim=3).squeeze(0).cpu().permute(
                       (1, 2, 0)).clamp(0, 1).numpy().astype(float))


if __name__ == '__main__':
    main()

