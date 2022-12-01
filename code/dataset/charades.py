import numpy as np
import h5py
import pandas as pd
import json
import os

from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import Dataset, DataLoader


def iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


class CharadesSTA(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, word2idx_path, annotation_path, v_feat_path, v_feat_path_vgg=None, max_video_seq_len=48,
                 subset=None, data_type='i3d'):
        self.data_type = data_type
        self.word2id = json.load((open(word2idx_path)))
        self.v_feat_path = v_feat_path
        self.v_feat_path_vgg = v_feat_path_vgg
        self.annotation_path = annotation_path
        self.max_video_seq_len = max_video_seq_len
        self.process_annotation(subset)

    def process_annotation(self, subset):
        if subset is not None:
            df = pd.read_pickle(self.annotation_path)[:subset]
        else:
            df = pd.read_pickle(self.annotation_path)
        num_clips = self.max_video_seq_len
        self.annos = []
        for i in tqdm(range(len(df)), total=len(df), desc='extracting captions data'):
            vid = df['video_name'][i]
            duration = df['duration'][i]
            timestamp = df['timestamps'][i]
            moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)], dtype=torch.float)

            sentence = df['sentence'][i]
            encoded_sentence = df['encoded_sentence'][i]

            iou2d = torch.ones(num_clips, num_clips)
            grids = iou2d.nonzero(as_tuple=False)
            candidates = grids * duration / num_clips
            iou2d = iou(candidates, moment).reshape(num_clips, num_clips)

            start_idx = torch.floor(moment[0] * num_clips / duration)
            end_idx = torch.ceil(moment[1] * num_clips / duration) - 1
            start_end_gt = torch.tensor([start_idx, end_idx]).long()  # 47 for the end index
            if start_end_gt[0] > start_end_gt[1]:
                print(start_idx, end_idx, moment)
            if start_end_gt[1] >= 48:
                start_end_gt[1] -= 1
            # candidates, moments are timestamp based
            start_end_offset = torch.ones(num_clips, num_clips, 2)  # not divided by number of clips
            start_end_offset[:, :, 0] = ((moment[0] - candidates[:, 0]) / duration).reshape(num_clips, num_clips)
            start_end_offset[:, :, 1] = ((moment[1] - candidates[:, 1]) / duration).reshape(num_clips, num_clips)

            mask2d_contrast = self.get_contrastive_mask(timestamp, duration)
            mask2d_contrast = torch.tensor(mask2d_contrast)

            self.annos.append({'vname': vid,
                               'sentence': sentence,
                               'encoded_sentence': encoded_sentence,
                               'timestamp': moment,
                               'iou2d': iou2d,
                               'start_end_offset': start_end_offset,
                               'start_end_gt': start_end_gt,
                               'duration': duration,
                               'mask2d_contrast': mask2d_contrast,
                               })

    def get_contrastive_mask(self, time_stamp, duration):
        num_clips = self.max_video_seq_len
        start_idx = int(np.floor(time_stamp[0] * num_clips / duration))
        end_idx = int(np.ceil(time_stamp[1] * num_clips / duration))

        x, y = np.arange(0, start_idx + 1., dtype=int), np.arange(end_idx - 1, num_clips, dtype=int)
        mask2d_pos = np.zeros((num_clips, num_clips), dtype=bool)
        mask_idx = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        mask2d_pos[mask_idx[:, 0], mask_idx[:, 1]] = 1

        mask2d_neg = np.zeros((num_clips, num_clips), dtype=bool)
        for offset in range(start_idx):
            i, j = range(0, start_idx - offset), range(offset, start_idx)
            mask2d_neg[i, j] = 1
        for offset in range(end_idx):
            i, j = range(end_idx, num_clips - offset), range(end_idx + offset, num_clips)
            mask2d_neg[i, j] = 1
        if np.sum(mask2d_neg) == 0:
            mask2d_neg[0, 0] = 1
            mask2d_neg[num_clips - 1, num_clips - 1] = 1
        return np.array([mask2d_pos, mask2d_neg])

    def pool_video(self, v_feat, max_seq_len=None):
        if max_seq_len == None:
            max_seq_len = self.max_video_seq_len
        feature_len = len(v_feat)
        segment_index = [[np.floor(i * feature_len / max_seq_len).astype(int),
                          np.ceil((i + 1) * feature_len / max_seq_len).astype(int)] for i in range(max_seq_len)]
        seg_fts = []
        for i, index in enumerate(segment_index):
            p_start = index[0]
            p_end = index[1]

            if (p_end - p_start) <= 1:
                ft_indices = [p_start]
            else:
                ft_indices = range(p_start, p_end)

            ft_indices = sorted(list(map(lambda x: min(feature_len - 1, x), ft_indices)))
            seg_feature = torch.max(v_feat[ft_indices, :], dim=0)[0]
            seg_fts.append(seg_feature)
        seg_fts = torch.stack(seg_fts)
        return seg_fts

    def average_to_fixed_length(self, v_feat, max_seq_len=None):
        if max_seq_len == None:
            max_seq_len = self.max_video_seq_len
            
        output = F.interpolate(v_feat.transpose(0, 1).unsqueeze(0),
                       size=max_seq_len, mode='linear',
                       align_corners=False)
        output = output[0, ...].transpose(0, 1)
        return output

    def pad_video(self, v_feat, max_seq_len=None):
        if max_seq_len == None:
            max_seq_len = self.max_video_seq_len
        v_len = v_feat.shape[0]
        feature_dim = v_feat.shape[1]
        need_pad = max_seq_len > v_len
        if need_pad:
            pad_len = max_seq_len - v_len
            pad_tensor = torch.zeros(size=(pad_len, feature_dim), device=v_feat.device)
            seg_fts = torch.cat((v_feat, pad_tensor), dim=0)
        else:
            seg_fts = self.pool_video(v_feat, max_seq_len=max_seq_len)
        return seg_fts

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        vname = self.annos[index]['vname']
        duration = self.annos[index]['duration']
        timestamp = self.annos[index]['timestamp']
        timestamp = torch.as_tensor(timestamp, dtype=torch.float32)
        text = torch.as_tensor(self.annos[index]['encoded_sentence'])
        sentence = self.annos[index]['sentence']
        iou2d = self.annos[index]['iou2d']
        mask2d_contrast = self.annos[index]['mask2d_contrast']
        start_end_offset = self.annos[index]['start_end_offset']
        start_end_gt = self.annos[index]['start_end_gt']

        if self.data_type == 'i3d':
            visual = np.load(os.path.join(self.v_feat_path, f"{vname}.npy"))
            visual = torch.FloatTensor(visual.squeeze())
        else:
            with h5py.File(self.v_feat_path_vgg, 'r') as f:
                visual = np.array(f[vname])
                visual = torch.FloatTensor(visual)
        nclips = visual.shape[0]  # number of clips

        visual = self.average_to_fixed_length(visual)
        visual_len = self.max_video_seq_len
        map_gt = np.zeros((2, visual_len), dtype=np.float32)
        gt_s, gt_e = start_end_gt.numpy()
        gt_length = gt_e - gt_s + 1  # make sure length > 0
        map_gt[0, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
        map_gt[1, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
        map_gt[0, map_gt[0, :] >= 0.8] = 1.
        map_gt[0, map_gt[0, :] < 0.1353] = 0.
        map_gt[1, map_gt[1, :] >= 0.8] = 1.
        map_gt[1, map_gt[1, :] < 0.1353] = 0.
        if (map_gt[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        map_gt = torch.from_numpy(map_gt)

        return visual, text, visual_len, iou2d, duration, timestamp, mask2d_contrast, \
               start_end_offset, start_end_gt, map_gt, vname, sentence


