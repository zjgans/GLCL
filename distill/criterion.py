from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

from .NCEAverage import NCEAverage
from .NCEAverage import NCESoftmax
from .NCECriterion import NCECriterion


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class NCELoss(nn.Module):
    """NCE contrastive loss"""
    def __init__(self, opt, n_data):
        super(NCELoss, self).__init__()
        self.contrast = NCEAverage(opt.feat_dim, n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = NCECriterion(n_data)
        self.criterion_s = NCECriterion(n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class NCESoftmaxLoss(nn.Module):
    """info NCE style loss, softmax"""
    def __init__(self, opt, n_data):
        super(NCESoftmaxLoss, self).__init__()
        self.contrast = NCESoftmax(opt.feat_dim, n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = nn.CrossEntropyLoss()
        self.criterion_s = nn.CrossEntropyLoss()

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        bsz = f_s.shape[0]
        label = torch.zeros([bsz, 1]).cuda().long()
        s_loss = self.criterion_s(out_s, label)
        t_loss = self.criterion_t(out_t, label)
        loss = s_loss + t_loss
        return loss


class Attention(nn.Module):
    """attention transfer loss"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, K=[]):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k

if __name__ == '__main__':
    pass
