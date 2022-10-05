import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
from tqdm import tqdm


def get_iou(pred, target, return_raw=True):
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]

    target_left = target[:, 0]
    target_right = target[:, 1]

    intersect = np.minimum(pred_right, target_right) - np.maximum(pred_left, target_left)
    intersect[intersect < 0] = 0
    target_area = target_right - target_left
    pred_area = pred_right - pred_left
    pred_area[pred_area < 0] = 0
    union = target_area + pred_area - intersect
    IOU = intersect / (union + 1e-8)

    assert IOU.size != 0
    if return_raw:
        return IOU
    else:
        return IOU.mean()


def nms(moments, scores, topk=5, thresh=0.5):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = torch.zeros_like(ranks).bool()
    numel = suppressed.numel()
    count = 0
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    return moments[~suppressed]


def iou(candidates, gt):
    '''
    candidates: (prop_num, 2)
    gt: (2, )
    '''
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def batch_iou(candidates, gt):
    '''
    candidates: (batch, prop_num, 2)
    gt: (batch, 2)
    '''
    bs, prop_num, _ = candidates.size()
    gt = gt.unsqueeze(1).repeat(1, prop_num, 1)

    start, end = candidates[:, :, 0], candidates[:, :, 1]  # batch, prop_num
    s, e = gt[:, :, 0].float(), gt[:, :, 1].float()  # batch, prop_num
    inter = end.min(e) - start.max(s)  # batch, prop_num
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero(as_tuple=False)
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def evaluate(score_pred, time_stamp, duration, nms_thresh=0.5, recall_metrics=[1, 5], iou_metrics=[0.3, 0.5, 0.7]):
    '''
    :param score_pred: confidence score (all_batch, seq, seq)
    :param time_stamp: (all_batch, 2)
    :param duration: (all_batch, 1) or (all_batch, )
    :param nms_thresh: non-maximum suppression threshold
    :param recall_metrics: R@1 and R@5
    :param iou_metrics: iou>0.3, 0.5, 0.7
    :return:
    '''
    device = score_pred.device
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics, device=device)
    iou_metrics = torch.tensor(iou_metrics, device=device)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics, device=device)
    num_clips = score_pred[0].shape[-1]
    for idx, score2d in tqdm(enumerate(score_pred)):
        candidates, scores = score2d_to_moments_scores(score2d, num_clips, duration[idx])
        moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)

        for i, r in enumerate(recall_metrics):
            mious = iou(moments[:r], time_stamp[idx])
            bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
            recall_x_iou[i] += bools.any(dim=0)
    recall_x_iou /= len(score_pred)
    return recall_x_iou


def evaluate_1d(score_pred, prop_s_e, time_stamp, duration, num_clips=48, nms_thresh=0.7, recall_metrics=[1, 5], iou_metrics=[0.3, 0.5, 0.7]):
    '''
    :param score_pred: confidence score (all_batch, prop_num)
    :param prop_s_e: (all_batch, prop_num, 2)
    :param time_stamp: (all_batch, 2)
    :param duration: (all_batch, 1) or (all_batch, )
    :param nms_thresh: non-maximum suppression threshold
    :param recall_metrics: R@1 and R@5
    :param iou_metrics: iou>0.3, 0.5, 0.7
    :return:
    '''
    device = score_pred.device
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics, device=device)
    iou_metrics = torch.tensor(iou_metrics, device=device)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics, device=device)
    top32 = 32
    mean_ious = []
    for idx, score1d in tqdm(enumerate(score_pred)):
        candidates = prop_s_e[idx] * duration[idx] / num_clips
        moments = nms(candidates, score1d, topk=recall_metrics[-1], thresh=nms_thresh)
        for i, r in enumerate(recall_metrics):
            mious = iou(moments[:r], time_stamp[idx])
            bools = mious[:, None].expand(-1, num_iou_metrics) > iou_metrics
            recall_x_iou[i] += bools.any(dim=0)
        mious = iou(moments[:top32], time_stamp[idx]).mean()
        mean_ious.append(mious)
    mean_ious = torch.stack(mean_ious)
    mean_ious = mean_ious.mean()
    recall_x_iou /= len(score_pred)
    return recall_x_iou, mean_ious

