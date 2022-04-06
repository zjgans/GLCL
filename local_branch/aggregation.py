import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import SupCluLoss


class Aggregation(nn.Module):
    def __init__(self, base_encoder,T=0.07):

        super(Aggregation, self).__init__()
        self.encoder = base_encoder
        self.criterion_clu = SupCluLoss(temperature=T)
        self.criterion_loc = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()

        # num_classes is the output fc dimension

    @torch.no_grad()
    def _montage_opera(self, images):

        n, c, h, w = images[0].shape
        permute = torch.randperm(n * 4).cuda()

        un_shuffle_permute = torch.argsort(permute)

        images_gather = torch.cat(images, dim=0)
        images_gather = images_gather[permute, :, :, :]
        col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
        col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2).cuda()

        return images_gather, permute, n,un_shuffle_permute

    @torch.no_grad()
    def angle_preprocess(self,v,class_weight):
        v = F.normalize(F.normalize(v,dim=1)-F.normalize(class_weight,dim=1),dim=1)
        return v

    def forward(self, images, targets=None):
            images_gather, permute, bs_all,un_shuffle_permute = self._montage_opera(images)

            # compute features
            q = self.encoder(images_gather,use_clu_loc=True)
            n, c, h, w = q.shape
            c1, c2 = q.split([1, 1], dim=2)
            f1, f2 = c1.split([1, 1], dim=3)
            f3, f4 = c2.split([1, 1], dim=3)
            q_gather = torch.cat([f1, f2, f3, f4], dim=0)
            q_gather = q_gather.view(n * 4, -1)

            # local contrastive
            q_local = q_gather[un_shuffle_permute]
            q_local = q_local.chunk(4)
            q_local = torch.cat(q_local, dim=1)
            local_inv_rep = self.encoder.local_head(q_local)

            train_targets = targets.repeat(4)
            train_targets = train_targets[permute]

            # clustering branch
            q_agg = self.encoder.local_identity_head(q_gather)
            q_agg = nn.functional.normalize(q_agg, dim=1)
            loss_agg = self.criterion_clu(q_agg, train_targets)

            # location branch
            label_loc = torch.LongTensor([0] * bs_all + [1] * bs_all + [2] * bs_all + [3] * bs_all).cuda()

            label_loc = label_loc[permute]
            logits_loc = self.encoder.fc_loc(q_gather)
            loss_loc = self.criterion_loc(logits_loc, label_loc)

            return (loss_agg,loss_loc),[q_agg,logits_loc,local_inv_rep]

