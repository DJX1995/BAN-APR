import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import contrast_selection
from evaluation import batch_iou


def sim(x, y):
    '''
    compute dot product similarity
    :param x: (1,  hidden)
    :param y: (batch,  hidden)
    '''
    normx = torch.linalg.norm(x, dim=-1)
    normy = torch.linalg.norm(y, dim=-1)
    x_norm = x / (normx.unsqueeze(-1) + 1e-8)
    y_norm = y / (normy.unsqueeze(-1) + 1e-8)
    return torch.matmul(x_norm, y_norm.T)


class ContrastLoss(nn.Module):
    def __init__(self, margin=1, tao=1., neg_ratio=20.):
        super().__init__()
        self.margin = margin
        self.tao = tao
        self.neg_ratio = neg_ratio

    def forward(self, pos_query, tmap, mask2d_pos, mask2d_neg):
        '''
        :param tmap: (batch, seq, seq, hidden)
        :param mask2d_pos: (batch, seq, seq)
        :param mask2d_neg: (batch, seq, seq)
        :param ious: (batch, seq, seq)
        :return:
        '''
        loss = []
        for i in range(tmap.size(0)):
            # masked selection
            tmp1 = tmap[i][mask2d_pos[i], :]
            tmp2 = tmap[i][mask2d_neg[i], :]
            pos = tmp1 / torch.linalg.norm(tmp1, dim=-1).unsqueeze(-1)
            neg = tmp2 / torch.linalg.norm(tmp2, dim=-1).unsqueeze(-1)
            if pos.size(0) == 0 or neg.size(0) == 0:
                continue
            positive_sim = sim(pos_query[i].unsqueeze(0), pos)  # (1, pos_num)
            negative_sim = sim(pos_query[i].unsqueeze(0), neg)  # (1, neg_num)
            all_sim = torch.cat([positive_sim, negative_sim], dim=-1)  # (1, neg_num+pos_num)
            numerator = torch.exp(positive_sim / self.tao)  # (1, pos_num)
            numerator = numerator.sum(dim=-1)
            denominator = torch.exp(all_sim / self.tao).sum(dim=-1)  # (1, 1)
            tmp = -torch.log(numerator / (denominator + 1e-8))
            loss.append(tmp)
        return sum(loss) / len(loss)


def temporal_difference_loss(td, position_mask):
    '''
    td: (bs, seq)
    position_mask: (bs, seq), smoothed scores for start/end confidence
    '''
    td = td.softmax(dim=-1)
    numerator = position_mask * torch.log(td)
    numerator = numerator.sum(dim=-1)
    denominator = position_mask.sum(dim=-1)
    loss = -numerator / (denominator + 1e-8)
    return loss.mean()
    

class MainLoss(nn.Module):
    def __init__(self, min_iou, max_iou, temperature=0.5):
        super().__init__()
        self.t = temperature
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.contrast_loss = ContrastLoss()
        self.boundary_loss = nn.CrossEntropyLoss(reduction='none')
        self.offset_loss = nn.SmoothL1Loss()

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def forward(self, out_feature, data, td):
        scores2d, ious2d, mask2d, s_e_distribution, mask2d_contrast, map2d_proj, \
        sen_proj, pred_s_e_round, final_pred, offset_pred, offset_gt = \
            out_feature['tmap'], data['iou2d'], out_feature['map2d_mask'], data['s_e_distribution'], \
            data['mask2d_contrast'], out_feature['map2d_proj'], out_feature['sen_proj'], \
            out_feature['coarse_pred_round'], out_feature['final_pred'], out_feature['offset'], \
            out_feature['offset_gt']
        ious2d_scaled = self.scale(ious2d).clamp(0, 1)
        # loss_bce = F.binary_cross_entropy_with_logits(
        #     scores2d.squeeze().masked_select(mask2d),
        #     ious2d.masked_select(mask2d),
        #     reduction='none')
        # loss_bce = loss_bce * torch.pow(torch.sigmoid(scores2d).squeeze().masked_select(mask2d) - ious2d_scaled.masked_select(mask2d), 2)
        # loss_bce = loss_bce.mean()
        loss_bce = F.binary_cross_entropy_with_logits(
            scores2d.squeeze().masked_select(mask2d),
            ious2d_scaled.masked_select(mask2d)
        )

        td_mask = s_e_distribution.sum(dim=1)
        loss_td = temporal_difference_loss(td, td_mask)

        mask2d_pos = mask2d_contrast[:, 0, :, :]
        mask2d_neg = mask2d_contrast[:, 1, :, :]
        mask2d_pos = torch.logical_and(mask2d, mask2d_pos)
        mask2d_neg = torch.logical_and(mask2d, mask2d_neg)
        loss_contrast = self.contrast_loss(sen_proj, map2d_proj, mask2d_pos, mask2d_neg)

        ious_gt = []
        for i in range(ious2d_scaled.size(0)):
            start = pred_s_e_round[i][:, 0]
            end = pred_s_e_round[i][:, 1] - 1
            final_ious = ious2d_scaled[i][start, end]
            ious_gt.append(final_ious)
        ious_gt = torch.stack(ious_gt)
        # loss_refine = F.binary_cross_entropy_with_logits(
        #     final_pred.squeeze(),
        #     ious_gt,
        #     reduction='none'
        # )
        # loss_refine = loss_refine * torch.pow(torch.sigmoid(final_pred).squeeze() - ious_gt, 2)
        # loss_refine = loss_refine.mean()
        loss_refine = F.binary_cross_entropy_with_logits(
            final_pred.squeeze().flatten(),
            ious_gt.flatten()
        )

        offset_pred = offset_pred.reshape(-1, 2)
        offset_gt = offset_gt.reshape(-1, 2)
        loss_offset = 0.
        loss_offset += self.offset_loss(offset_pred[:, 0], offset_gt[:, 0])
        loss_offset += self.offset_loss(offset_pred[:, 1], offset_gt[:, 1])

        loss = {'loss_bce': loss_bce,
                'loss_refine': loss_refine,
                'loss_td': loss_td,
                'loss_contrast': loss_contrast,
                'loss_offset': loss_offset
                }
        return loss
    
