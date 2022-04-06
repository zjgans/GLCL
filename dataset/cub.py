import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUB(Dataset):

    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,transform=None):
        super(Dataset, self).__init__()

        self.normlize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        IMAGE_PATH = os.path.join(args.data_root,'images')
        SPLIT_PATH = os.path.join(args.data_root, 'split/')
        txt_path = osp.join(SPLIT_PATH, partition + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []
        self.args = args
        if partition == 'train':
            lines.pop(5864)  #this image file is broken

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, wnid+'/'+name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        self.imgs = data
        self.labels = label
        self.classes = np.unique(np.array(label)).shape[0]

        img_label = []
        for id, (img, label) in enumerate(zip(self.imgs, self.labels)):
            img_label.append((img, label))
        self.img_label = img_label

        if partition == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normlize])
        else:

            image_size = 84
            resize_size = 92
            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normlize])

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):

        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class MetaCub(CUB):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaCub, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        # self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):

            support_xs_ids_sampled = np.random.choice(len(self.data[cls]), self.n_shots + self.n_queries, False)
            indexDtrain = np.array(support_xs_ids_sampled[:self.n_shots])
            indexTest = np.array(support_xs_ids_sampled[self.n_shots:])
            support_xs.append(np.array(self.data[cls])[indexDtrain].tolist())
            support_ys.append([idx] * self.n_shots)
            query_xs.append((np.array(self.data[cls])[indexTest]).tolist())
            query_ys.append([idx] * self.n_queries)

        support_xs = [img_path for cls in support_xs for img_path in cls]
        query_xs = [img_path for cls in query_xs for img_path in cls]

        if self.n_aug_support_samples > 1:
            support_xs = np.array(support_xs)
            support_ys = np.array(support_ys)
            support_xs = np.tile(support_xs, (self.n_aug_support_samples))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))

        support_xs = torch.stack(list(map(lambda x: self.test_transform(Image.open(x).convert('RGB')), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(Image.open(x).convert('RGB')), query_xs)))

        support_ys, query_ys = torch.tensor(support_ys), torch.tensor(query_ys)

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs