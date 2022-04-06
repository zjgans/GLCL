from __future__ import print_function

import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.cub import CUB,MetaCub
from dataset.transform_cfg import transforms_options, transforms_test_options

def get_dataloaders(opt):
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'

    if opt.dataset == 'miniImageNet':

        train_trans, test_trans = transforms_options[opt.transform]
        dataset = ImageNet(args=opt, partition=train_partition, transform=train_trans)
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

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

        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

        no_sample = len(ImageNet(args=opt, partition=train_partition, transform=train_trans))

    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = TieredImageNet(args=opt, partition=train_partition, transform=train_trans)
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

        no_sample = len(TieredImageNet(args=opt, partition=train_partition, transform=train_trans))

    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = CIFAR100(args=opt, partition=train_partition, transform=train_trans)

        # sampler = RandomIdentitySampler(dataset.img_label,opt.num_instances)
        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)

        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]

        meta_trainloader = DataLoader(MetaCIFAR100(args=opt, partition='train',
                                                   train_transform=train_trans,
                                                   test_transform=test_trans),
                                      batch_size=1, shuffle=True, drop_last=False,
                                      num_workers=opt.num_workers)

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

        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
        no_sample = len(CIFAR100(args=opt, partition=train_partition, transform=train_trans))

    elif opt.dataset == 'cub':
        opt.transform = 'E'
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = CUB(args=opt, partition=train_partition, transform=train_trans)
        train_loader = DataLoader(CUB(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CUB(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]

        meta_testloader = DataLoader(MetaCub(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCub(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)

        n_cls = 100

        no_sample = len(CUB(args=opt, partition=train_partition, transform=train_trans))

    else:
        raise NotImplementedError(opt.dataset)

    return val_loader, meta_testloader, meta_valloader, n_cls, no_sample, dataset

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class JigCluTransform:
    def __init__(self, transform, opt):
        self.transform = transform
        self.c = opt.cross_ratio
        self.opt = opt

    def __call__(self, x):

        if self.opt.dataset == 'cub':

            x = Image.open(x).convert('RGB')
        else:
            x = Image.fromarray(x)
        h, w = x.size

        ch = self.c * h
        cw = self.c * w

        # x = transforms.RandomCrop(32,padding=4)(x)
        if self.opt.dataset =='CIFAR-FS' or self.opt.dataset=='FC100':
            x = transforms.RandomCrop(32,padding=4)(x)

            return [
                    self.transform(transforms.functional.resized_crop(x,0,0,h//2+ch,w//2+cw,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,0, w//2-cw,  h//2+ch,w,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,h//2-ch,0, h, w//2+cw,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,h//2-ch, w//2-cw,  h, w,(32,32)))]
        # else:
        #     x = transforms.RandomCrop(84,padding=8)(x)
        #     return [
        #         self.transform(transforms.functional.resized_crop(x, 0, 0, h // 2 + ch, w // 2 + cw, (42,42))),
        #         self.transform(transforms.functional.resized_crop(x, 0, w // 2 - cw, h // 2 + ch, w, (42,42))),
        #         self.transform(transforms.functional.resized_crop(x, h // 2 - ch, 0, h, w // 2 + cw, (42,42))),
        #         self.transform(transforms.functional.resized_crop(x, h // 2 - ch, w // 2 - cw, h, w, (42,42)))]
        else:
            return [self.transform(x.crop((0, 0, h // 2 + ch, w // 2 + cw))),
                        self.transform(x.crop((0, w // 2 - cw, h // 2 + ch, w))),
                        self.transform(x.crop((h // 2 - ch, 0, h, w // 2 + cw))),
                        self.transform(x.crop((h // 2 - ch, w // 2 - cw, h, w)))]


class GeomTransform:

    def __init__(self, normalize, opt):

        self.normalize = normalize
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.opt = opt
        self.size = opt.crop_size

    def transform_sample(self, img, indx=None):
        if indx is not None:
            out = transforms.functional.resized_crop(img, indx[0], indx[1], indx[2], indx[3], (self.size, self.size))
        else:
            out = img
        out = self.color_transform(out)
        # out = transforms.RandomGrayscale(p=0.2)(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __call__(self, x):
        # img = np.asarray(x).astype('uint8')
        if self.opt.dataset == 'CIFAR-FS' or self.opt.dataset == 'FC100':

            img = transforms.RandomCrop(32, padding=4)(Image.fromarray(x))
            img2 = self.transform_sample(img, [np.random.randint(10), 0, 22, 32])
            img3 = self.transform_sample(img, [0, np.random.randint(10), 32, 22])
            img4 = self.transform_sample(img, [np.random.randint(10), np.random.randint(10), 22, 22])
            img = self.transform_sample(img)

        if self.opt.dataset == 'miniImageNet':

            img = transforms.RandomCrop(84, padding=8)(Image.fromarray(x))
            img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
            img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
            img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])
            img = self.transform_sample(img)
        if self.opt.dataset == 'tieredImageNet':
            img = transforms.RandomCrop(84, padding=8)(Image.fromarray(x))
            img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
            img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
            img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])
            img = self.transform_sample(img)
        if self.opt.dataset == 'cub':
            img = transforms.RandomCrop(84, padding=8)(Image.open(x).convert('RGB'))
            img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
            img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
            img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])
            img = self.transform_sample(img)

        return [img,img2,img3,img4]

def get_jigclu_dataloader(opt, dataset):
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([

            # transforms.Resize((32,32)),
            # transforms.RandomCrop(32,padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(dataset, JigCluTransform(trans, opt), GeomTransform(normalize, opt)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True)

    if opt.dataset == 'miniImageNet':
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(dataset, JigCluTransform(trans, opt), GeomTransform(normalize, opt)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers,
                                  )

    if opt.dataset == 'tieredImageNet':
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(dataset, JigCluTransform(trans, opt), GeomTransform(normalize, opt)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True
                                  )
    if opt.dataset == 'cub':
        normalize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        trans = transforms.Compose([
            # lambda x :Image.open(x).convert('RGB'),
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        train_loader = DataLoader(Preprocessor(dataset, JigCluTransform(trans, opt), GeomTransform(normalize, opt)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True
                                  )

    return train_loader

class Preprocessor(Dataset):
    def __init__(self, dataset, jig_transforms=None, geo_transforms=None):
        super(Preprocessor, self).__init__()
        self.jig_transforms = jig_transforms
        self.geo_transforms = geo_transforms
        self.label_transform = torch.LongTensor()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]

        img_jig = self.jig_transforms(img)
        img_geo = self.geo_transforms(img)

        return img_jig, img_geo, label, item
