import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def simple_contrstive_loss(vi_batch, vi_t_batch, mn_arr, temp_parameter=0.1):
    """
    Returns the probability that feature representation for image I and I_t belong to same distribution.
    :param vi_batch: Feature representation for batch of images I
    :param vi_t_batch: Feature representation for batch containing transformed versions of I.
    :param mn_arr: Memory bank of feature representations for negative images for current batch
    :param temp_parameter: The temperature parameter
    """

    # Define constant eps to ensure training is not impacted if norm of any image rep is zero
    eps = 1e-6

    # L2 normalize vi, vi_t and memory bank representations
    vi_norm_arr = torch.norm(vi_batch, dim=1, keepdim=True)
    vi_t_norm_arr = torch.norm(vi_t_batch, dim=1, keepdim=True)
    mn_norm_arr = torch.norm(mn_arr, dim=1, keepdim=True)

    vi_batch = vi_batch / (vi_norm_arr + eps)
    vi_t_batch = vi_t_batch/ (vi_t_norm_arr + eps)
    mn_arr = mn_arr / (mn_norm_arr + eps)

    # Find cosine similarities
    sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_t_mn_mat = (vi_t_batch @ mn_arr.t())

    # Fine exponentiation of similarity arrays
    exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = torch.sum(exp_sim_vi_t_mn_mat, 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr + eps)

    neg_log_img_pair_probs = -1 * torch.log(batch_prob_arr)
    loss_i_i_t = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]
    return loss_i_i_t



class ContrastByClassCalculator(nn.Module):
    def __init__(self,T,num_classes,trans,num_samples,momentum,dim=128,K=4096):
        super(ContrastByClassCalculator, self).__init__()
        self.T =T
        self.K =K
        self.trans = trans
        self.num_classes=num_classes
        self.m = momentum
        self.register_buffer("queue",torch.randn(num_classes,dim,K))
        self.queue = F.normalize(self.queue,dim=1)
        self.register_buffer("queue_ptr",torch.zeros(num_classes,dtype=torch.long))


    def forward(self,q,k,weight,cls_labels,criterion,use_angle=False):
        queue = self.queue[:self.num_classes,:,:].detach().clone()
        # self.dequeue_and_enqueue(k,cls_labels)

        fin_weight =weight.unsqueeze(2)
        if use_angle:
            class_weight_by_label = weight[cls_labels]
            q = self.angle_preprocess(q,class_weight=class_weight_by_label)
            k = self.angle_preprocess(k,class_weight=class_weight_by_label)
            queue = self.angle_preprocess(queue,class_weight=fin_weight)
        l_pos = torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1)

        labels_onehot= torch.zeros((cls_labels.size(0),self.num_classes)).cuda().scatter(
            1,cls_labels.unsqueeze(1),1)
        q_onehot = labels_onehot.unsqueeze(-1) * q.unsqueeze(1)
        l_neg = torch.einsum('ncd,cdk->nk',q_onehot,queue)

        logits = torch.cat([l_pos,l_neg],dim=1)
        logits /=self.T
        labels = torch.zeros(logits.size(0),dtype=torch.long).cuda()

        loss = criterion(logits,labels)

        return loss

    @ torch.no_grad()
    def dequeue_and_enqueue(self,keys,cls_label):
        for cls_id in torch.unique(cls_label):
            cls_keys = keys[cls_label==cls_id]
            num_keys = cls_keys.size(1)
            batch_size = cls_keys.size(0)
            ptr = int(self.queue_ptr[cls_id])

            if ptr + batch_size >= self.K:
                self.queue[cls_id][:,ptr:]= cls_keys.T[:,:self.K - ptr]
                self.queue[cls_id][:,:(ptr+batch_size) % self.K] = cls_keys.T[:,self.K-ptr:]
            else:
                self.queue[cls_id][:,ptr:ptr + batch_size] =cls_keys.T
            ptr = (ptr + batch_size)%self.K
            self.queue_ptr[cls_id]=ptr

    def angle_preprocess(self,v,class_weight):
        v = F.normalize(F.normalize(v)-F.normalize(class_weight))
        return v

