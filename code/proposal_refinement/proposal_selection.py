import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import math
from evaluation import iou


def proposal_selection_with_negative(moments, scores, thresh=0.5, topk=5, neighbor=16, negative=16):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = torch.zeros_like(ranks).bool()
    select = torch.zeros_like(ranks).bool()
    numel = suppressed.numel()
    count = 0
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i] = True
        select[i] = True
        ind_sel = mask.nonzero(as_tuple=False).squeeze(-1)
        if ind_sel.numel() != 0:
            #             suppressed[i + 1:][ind_sel] = True
            ind_sel = ind_sel[:neighbor]
            select[i + 1:][ind_sel] = True
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    total_num = topk * (neighbor + 1)
    if select.sum() < total_num:
        moments_sel_pos = moments[~suppressed][:int(total_num - select.sum())]
        moments_sel_neg = torch.flip(moments[~suppressed], dims=(0,))[:negative]
        moments_sel = torch.cat([moments_sel_neg, moments_sel_pos, moments[select]], dim=0)
    else:
        moments_sel_neg = torch.flip(moments[~suppressed], dims=(0,))[:negative]
        moments_sel = torch.cat([moments_sel_neg, moments[select]], dim=0)
    return moments_sel


class Aaptive_Proposal_Sampling(nn.Module):
    def __init__(self, topk=5, neighbor=16, negative=16, thresh=0.5):
        super().__init__()
        self.topk = topk
        self.neighbor = neighbor
        self.thresh = thresh
        self.negative = negative

    def forward(self, score_pred, map2d_mask, map2d, offset_gt, tmap):
        pred_s_e = []
        prop_lists = []
        offset_gt_list = []
        pred_score = []
        for b in range(score_pred.size(0)):
            grids = map2d_mask.nonzero(as_tuple=False)
            scores = score_pred[b][grids[:, 0], grids[:, 1]]
            grids[:, 1] += 1
            prop_s_e_topk = proposal_selection_with_negative(grids, scores,
                                                             thresh=self.thresh,
                                                             topk=self.topk,
                                                             neighbor=self.neighbor,
                                                             negative=self.negative)
            segs = map2d[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1]
            prop_lists.append((segs))
            offset_gt_list.append(offset_gt[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1, :])
            pred_s_e.append(prop_s_e_topk)
            pred_score.append(tmap[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1])
        prop_feature = torch.cat(prop_lists, dim=0)  # (bs x prop_num, dim)
        pred_s_e = torch.cat(pred_s_e, dim=0)  # (bs x prop_num, 2)
        offset_gt = torch.cat(offset_gt_list, dim=0)  # (bs x prop_num, 2)
        pred_score = torch.cat(pred_score, dim=0)  # (bs x prop_num, 2)
        return prop_feature, pred_s_e, offset_gt, pred_score