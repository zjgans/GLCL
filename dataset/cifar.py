from __future__ import print_function

import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class CIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None,recompute=False):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.pretrain = pretrain
        self.simclr = args.simclr

        if self.pretrain:
            self.file_pattern = '%s.pickle'
        else:
            self.file_pattern = '%s.pickle'
        self.data = {}

        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for idx, label in enumerate(labels):
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = []
            for idx, label in enumerate(labels):
                new_labels.append(label2label[label])
            self.labels = new_labels

        img_label=[]
        for id,(img,label) in  enumerate(zip(self.imgs,self.labels)):
            img_label.append((img,label))
        self.img_label=img_label
        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def transform_sample(self, img, indx=None):
        if indx is not None:
            out = transforms.functional.resized_crop(img, indx[0], indx[1], indx[2], indx[3], (32,32))
        else:
            out = img
        out = self.color_transform(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __getitem__(self, item):
        img,target,_ = self.img_label[item]
        img = np.asarray(img).astype('uint8')
        # target = self.labels[item] - min(self.labels)

        if self.partition == 'train':
            img = transforms.RandomCrop(32, padding=4)(Image.fromarray(img))
        else:
            img = Image.fromarray(img)

        img2 = self.transform_sample(img, [np.random.randint(10), 0, 22, 32])
        img3 = self.transform_sample(img, [0, np.random.randint(10), 32, 22])
        img4 = self.transform_sample(img, [np.random.randint(10), np.random.randint(10), 22, 22])

        if self.partition == 'train':
            img = self.transform_sample(img)
        else:
            img = transforms.functional.to_tensor(img)
            img = self.normalize(img)

        if not self.is_sample:
            return [img,img2,img3,img4],target
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)
    
class MetaCIFAR100(CIFAR100):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaCIFAR100, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        
        support_xs = []
        support_ys = []
        support_ts = []
        query_xs = []
        query_ys = []
        query_ts = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            support_ts.append([cls] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
            query_ts.append([cls] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        support_ts, query_ts = np.array(support_ts), np.array(query_ts)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))
        query_ts = query_ts.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
            support_ts = np.tile(support_ts.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        
        
        
        query_xs = query_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            query_xs = np.tile(query_xs, (self.n_aug_support_samples, 1, 1, 1))
            query_ys = np.tile(query_ys.reshape((-1,)), (self.n_aug_support_samples))
            query_ts = np.tile(query_ts.reshape((-1,)), (self.n_aug_support_samples))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs

class ClusterCIFAR100(CIFAR100):
    def __init__(self,args,partition='train', transforms=None):
        super(ClusterCIFAR100,self).__init__()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_label)
    def __getitem__(self, item):

        img,label,fname = self.img_label(item)
        img = self.transforms(img)

        return img,label,fname,item

if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = '/dataset/FC100'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = CIFAR100(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaCIFAR100(args, 'train')
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
