import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import numpy as np


class DenseMaxPool(nn.Module):
    def __init__(self, N, device='cpu'):
        super().__init__()
        self.identity = nn.Identity()
        self.pool = nn.MaxPool1d(2, stride=1)
        self.seq_len = N
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1
        maskij = []
        for idx in range(N):
            start_idxs = [s_idx for s_idx in range(0, N - idx, 1)]
            end_idxs = [s_idx + idx for s_idx in start_idxs]
            mask2d[start_idxs, end_idxs] = 1
            # mask2d[:, :, start_idxs, end_idxs] = 1
            maskij.append((start_idxs, end_idxs))
        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        for idx in range(self.seq_len):
            if idx == 0:
                x = self.identity(x)
            else:
                x = self.pool(x)
            start_idxs, end_idxs = self.maskij[idx]
            map2d[:, :, start_idxs, end_idxs] = x
        return map2d, self.mask2d


class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N, device='cpu'):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset), range(offset, N)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for i in range(1, len(pooling_counts)):
            poolers.extend(
                [nn.MaxPool1d(2 * i + 1, 1) for _ in range(pooling_counts[i])]
            )
        self.mask2d = mask2d
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d, self.mask2d


class SparseBoundaryCat(nn.Module):
    def __init__(self, pooling_counts, N, device='cpu'):
        super(SparseBoundaryCat, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset), range(offset, N)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for i in range(1, len(pooling_counts)):
            poolers.extend(
                [nn.MaxPool1d(2 * i + 1, 1) for _ in range(pooling_counts[i])]
            )
        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, start, end):
        B, D, N = start.shape
        map2d = start.new_zeros(B, 2 * D, N, N)
        map2d[:, :, range(N), range(N)] = torch.cat([start, end], dim=1)
        for (i, j) in self.maskij:
            tmp = torch.cat((start[:, :, i], end[:, :, j]), dim=1)
            map2d[:, :, i, j] = tmp
        return map2d, self.mask2d


class Aggregation_center(nn.Module):
    def __init__(self, config, device):
        super(Aggregation_center, self).__init__()
        hidden = config.vilt.fuse_dim
        max_video_seq_len = config.model.video_seq_len
        self.content_aggregation = SparseMaxPool(config.model.pooling_counts, max_video_seq_len, device)
        # self.content_aggregation = DenseMaxPool(max_video_seq_len, device)

    def forward(self, hidden_c):
        map2d_c, map2d_mask = self.content_aggregation(hidden_c.permute(0, 2, 1))
        map2d_c = map2d_c.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        return map2d_c, map2d_mask


