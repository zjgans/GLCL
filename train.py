from __future__ import print_function

import os
import argparse
import socket
import time
import wandb
import sys
from tqdm import tqdm
import mkl
import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from dataset.transform_cfg import transforms_options, transforms_list
from models import model_pool
from models.util import create_model
from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, Logger, generate_final_report, \
    restart_from_checkpoint,data_write_csv
from eval.meta_eval import meta_test, meta_test_tune
from eval.cls_eval import validate

from local_branch.aggregation import Aggregation

from losses import simple_contrstive_loss
from dataloader import get_dataloaders,get_jigclu_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] ='0'
os.environ["CUDA_LAUNCH_BLOCKING"]='0'
mkl.set_num_threads(2)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=5, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=5, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=65, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--ssl', type=bool, default=True, help='use self supervised learning')
    parser.add_argument('--tags', type=str, default="gen0, ssl", help='add tags for the experiment')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100','cub'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='ckpt_epoch_50.pth')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='/data/~/dataset ', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')

    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='test', help='the experiment id')

    # hyper parameters
    parser.add_argument('--gamma', type=float, default=1, help='loss coefficient for local loss')
    parser.add_argument('--alpha', type=float, default=0.1, help='loss coefficient for knowledge distillation loss')
    parser.add_argument('--beta', type=float, default=1,help= 'loss coefficient for eq loss')
    parser.add_argument('--eta', type=float, default=1, help= ' loss coefficient for mutual loss')

    parser.add_argument('--membank_size', type=int, default=6400, help=' membank size for contrastive ssl loss')
    parser.add_argument('--proj_dim',type=float,default=128)
    parser.add_argument('--cross-ratio', default=0.2, type=float,help='four patches crop cross ratio')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--local_t', type=float, default=0.2, help='temperature for global contrastive loss')
    parser.add_argument('--mvavg_rate', type=float, default=0.99, help='temperature for contrastive ssl loss')
    parser.add_argument('--trans', type=int, default=4, help='number of transformations')

    opt = parser.parse_args()
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
        opt.crop_size = 32

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay,
                                                            opt.transform)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    # extras
    opt.fresh_start = True
    return opt

def main():
    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()

    val_loader, meta_testloader, meta_valloader, n_cls, no_sample,dataset = get_dataloaders(opt)
    train_cluster_loader = get_jigclu_dataloader(opt, dataset.img_label)

    # model
    model = create_model(opt.model, n_cls, opt.dataset, n_trans=opt.trans, embd_sz=opt.proj_dim)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # optimizer
    if opt.adam:
        print("Adam")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        print("SGD")
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    to_restore = {'epoch': 0}
    if opt.use_resume:
        print('------load the parameters from  checkpoint--------')
        restart_from_checkpoint(
            os.path.join(opt.save_folder, opt.resume_file),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
        )
    start_epoch = to_restore['epoch']
    multibranch_model = Aggregation(model).cuda()

    MemBank = np.random.randn(no_sample,opt.proj_dim)
    MemBank = torch.tensor(MemBank,dtype=torch.float).cuda()
    MemBankNorm = torch.norm(MemBank,dim=1,keepdim=True)
    MemBank = MemBank / (MemBankNorm + 1e-6)

    for epoch in range(start_epoch, opt.epochs+1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        loss,MemBank = train(epoch, train_cluster_loader, multibranch_model,criterion,optimizer,opt,MemBank)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluate
        start = time.time()
        feat_meta_test_acc, feat_meta_test_std = meta_test(model, meta_testloader, is_feat=True)
        test_time = time.time() - start
        print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
                                                                                            feat_meta_test_std,
                                                                                            test_time))
        # regular saving
        if epoch % opt.save_freq == 0 or epoch == opt.epochs:

            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # final report
    print("GENERATING FINAL REPORT")
    generate_final_report(model, opt, wandb)

def train(epoch, train_loader, model,criterion,optimizer, opt,MemBank):
    """One epoch training"""

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    loss3 = AverageMeter()
    loss4 = AverageMeter()
    loss5 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_indices = list(range(len(MemBank)))

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for iter, (jig_output,geo_inputs,targets,indices) in enumerate(pbar):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                targets = targets.cuda()
                input1 = geo_inputs[0].cuda()

            batch_size = jig_output[0].shape[0]

            generated_data = rotrate_concat([input1])
            train_targets = targets.repeat(opt.trans)

            # ===================memory bank of negatives for current batch ==========================
            np.random.shuffle(train_indices)
            mn_indices_all = np.array(list(set(train_indices) - set(indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:opt.membank_size]
            mn_arr = MemBank[mn_indices]
            mem_rep_of_batch_imgs = MemBank[indices]

            (loss_agg, loss_loc), local_out = model(jig_output, targets)
            _,(train_logits,inv_rep,eq_logits) = model.encoder(generated_data,inductive=True)

            loss_ce = criterion(train_logits,train_targets)

            inv_rep_0 = inv_rep[:batch_size,:]
            # #
            inv_local = local_out[-1]
            loss_mut = simple_contrstive_loss(inv_rep_0,inv_local,mn_arr,opt.local_t)
            for ii in range(1,opt.trans):
                loss_mut += simple_contrstive_loss(inv_local,inv_rep[(ii * batch_size):((ii+1)*batch_size),:],mn_arr,opt.local_t)
            loss_mut = loss_mut/(opt.trans)

            proxy_labels = torch.zeros(opt.trans * batch_size).cuda().long()
            for ii in range(opt.trans):
                proxy_labels[ii * batch_size:(ii + 1) * batch_size] = ii
            loss_eq = criterion(eq_logits,proxy_labels)

            # loss = loss_ce + loss_eq*opt.beta + loss_mut*opt.eta + (loss_agg + loss_loc) *opt.gamma
            loss = loss_ce + (loss_agg + loss_loc) *opt.gamma

            acc1, acc5 = accuracy(train_logits, train_targets,topk=(1, 5))

            losses.update(loss.item(), batch_size)
            loss1.update(loss_ce.item(),batch_size)
            loss2.update(loss_mut.item(),batch_size)
            loss3.update(loss_loc.item(), batch_size)
            loss4.update(loss_eq.item(),batch_size)
            loss5.update(loss_agg.item(),batch_size)

            top1.update(acc1[0], generated_data.size(0))
            top5.update(acc5[0], generated_data.size(0))

            # ====================update memory bank ==============================
            MemBankCopy =MemBank.clone().detach()
            MemBankCopy[indices] = (opt.mvavg_rate * MemBankCopy[indices]) + ((1-opt.mvavg_rate)* inv_rep_0)
            MemBank = MemBankCopy.clone().detach()

            # ===================backward=====================
            # for optimizer in optimizers:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({  "Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()),
                "Acc@5":'{0:.2f}'.format(top5.avg.cpu().numpy(),2),
                "Loss": '{0:.2f}'.format(losses.avg, 2),
                "loss_ce": '{0:.2f}'.format(loss1.avg, 2),
                "loss_mut": "{0:.2f}".format(loss2.avg, 2),
                "loss_loc": "{0:.2f}".format(loss3.avg, 2),
                "loss_eq": "{0:.2f}".format(loss4.avg,2),
                "loss_agg": "{0:.2f}".format(loss5.avg, 2),
            })

    return loss,MemBank

if __name__ == '__main__':
    main()