import torch
from torch import nn
import numpy as np
import math
from copy import deepcopy
import torch.nn.functional as F


class LearnPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(LearnPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.pos_embed.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q):
        bsz_q, d_model, q_frm = q.shape
        assert q_frm == self.pos_embed.weight.shape[0], (q_frm, self.pos_embed.weight.shape)
        q_pos = self.pos_embed.weight.clone()
        q_pos = q_pos.unsqueeze(0)
        q_pos = q_pos.expand(bsz_q, q_frm, d_model).transpose(1, 2)
        # q_pos = q_pos.contiguous().view(bsz_q, q_frm, n_head, d_k)
        q = q + q_pos
        return self.dropout(q)


# dynamic graph from knn
def knn(x, y=None, k=5):
    if y is None:
        y = x
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)
    return idx


def get_graph_feature(x, prev_x=None, k=5, idx_knn=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx_knn + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, K, D)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # (B, D, N, K)
    return feature


def dense_graph_feature(x, idx_knn):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    # tmp = torch.arange(0, num_points, device=x.device)
    # tmp = tmp[None, None, :]
    # tmp = tmp.repeat(batch_size, num_points, 1)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx_knn + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, num_points, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, num_points, 1)  # (B, N, K, D)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # (B, D, N, K)
    return feature


def adaptive_graph_feature(x):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    x = x.transpose(2, 1).contiguous()
    tmp1 = x.unsqueeze(1).repeat(1, num_points, 1, 1)
    tmp2 = x.unsqueeze(2).repeat(1, 1, num_points, 1)
    tmp3 = tmp1 - tmp2
    # x = x.view(batch_size, -1, num_points)
    # # idx_knn = torch.arange(0, num_points, device=x.device)
    # # idx_knn = idx_knn[None, None, :]
    # # idx_knn = idx_knn.repeat(batch_size, num_points, 1)
    # idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    # idx = (idx_knn + idx_base).view(-1)
    # _, num_dims, _ = x.size()
    # x = x.transpose(2, 1).contiguous()
    # feature = x.view(batch_size * num_points, -1)[idx, :]
    # feature = feature.view(batch_size, num_points, num_points, num_dims)
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, num_points, 1)  # (B, N, K, D)
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # (B, D, N, K)
    feature = torch.cat((tmp3, tmp2), dim=3).permute(0, 3, 1, 2)
    return feature


class GCNeXtBlock(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, groups=32, width_group=4):
        super(GCNeXtBlock, self).__init__()
        self.k = k
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        )  # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=(1, self.k), groups=groups, padding=(0, (self.k - 1) // 2)),
            nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # semantic graph

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f = get_graph_feature(x, k=self.k)  # (B, D, N, K)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]

        out = tout + 2 * identity + sout
        return self.relu(out)


class GCNeXtMoudle(nn.Module):
    def __init__(self, channel_in, channel_out, k_num, groups, width_group):
        super(GCNeXtMoudle, self).__init__()

        self.backbone = nn.Sequential(
            GCNeXtBlock(channel_in, channel_out, k_num, groups, width_group),
        )

    def forward(self, x):
        gcnext_feature = self.backbone(x)
        return gcnext_feature


class GCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.gcn.k
        self.fc = nn.Sequential(
            nn.Conv2d(2 * config.gcn.hidden_size, config.gcn.hidden_size, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x_f = get_graph_feature(x, k=self.k)  # (B, D, N, K)
        out = self.fc(x_f)  # edge convolution on semantic graph
        out = out.max(dim=-1, keepdim=False)[0]
        return out


class DenseGCN_weighted(nn.Module):
    def __init__(self, config, idx_knn):
        super().__init__()
        self.idx_knn = idx_knn
        self.fc = nn.Sequential(
            nn.Conv2d(2 * config.gcn.hidden_size, config.gcn.hidden_size, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x, weights):
        '''
        x: (B, N, D)
        weights: (B, N, N)
        '''
        B = x.size(0)
        x_f = dense_graph_feature(x, idx_knn=self.idx_knn.repeat(B, 1, 1))  # (B, D, N, N)
        out = self.fc(x_f)  # edge convolution on semantic graph
        out = out.max(dim=-1, keepdim=False)[0]
        return out


class DenseGCN(nn.Module):
    def __init__(self, config, idx_knn):
        super().__init__()
        self.idx_knn = idx_knn
        self.fc = nn.Sequential(
            nn.Conv2d(2 * config.gcn.hidden_size, config.gcn.hidden_size, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        B = x.size(0)
        x_f = dense_graph_feature(x, idx_knn=self.idx_knn.repeat(B, 1, 1))  # (B, D, N, N)
        out = self.fc(x_f)  # edge convolution on semantic graph
        out = out.max(dim=-1, keepdim=False)[0]
        return out


class AdaptiveGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(2 * config.gcn.hidden_size, config.gcn.hidden_size, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        B = x.size(0)
        x_f = adaptive_graph_feature(x)  # (B, D, N, N)
        out = self.fc(x_f)  # edge convolution on semantic graph
        out = out.max(dim=-1, keepdim=False)[0]
        return out


class Prop_Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gcn_layer = nn.ModuleList([deepcopy(GCN(config)) for _ in range(config.gcn.num_blocks)])

    def forward(self, prop_feature):
        # (B, N, D) -> (B, D, N)
        prop_feature = prop_feature.transpose(1, 2)
        # encode
        for layer in self.gcn_layer:
            prop_feature = layer(prop_feature)
        return prop_feature.transpose(1, 2)  # (B, D, N) -> (B, N, D)


class Dense_Prop_Interaction(nn.Module):
    def __init__(self, config, prop_num, device='cpu'):
        super().__init__()
        idx_knn = torch.stack([torch.arange(0, prop_num, device=device) for _ in range(prop_num)])
        self.gcn_layer = nn.ModuleList([deepcopy(DenseGCN(config, idx_knn))
                                        for _ in range(config.gcn.num_blocks)])

    def forward(self, prop_feature):
        # (B, N, D) -> (B, D, N)
        prop_feature = prop_feature.transpose(1, 2)
        # encode
        for layer in self.gcn_layer:
            prop_feature = layer(prop_feature)
        return prop_feature.transpose(1, 2)  # (B, D, N) -> (B, N, D)


class Adaptive_Prop_Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gcn_layer = nn.ModuleList([deepcopy(AdaptiveGCN(config))
                                        for _ in range(config.gcn.num_blocks)])

    def forward(self, prop_feature):
        # (B, N, D) -> (B, D, N)
        prop_feature = prop_feature.transpose(1, 2)
        # encode
        for layer in self.gcn_layer:
            prop_feature = layer(prop_feature)
        return prop_feature.transpose(1, 2)  # (B, D, N) -> (B, N, D)


if __name__ == "__main__":
    # from charades_config import config
    # prop_feature = torch.randn(size=(32, 48, 512))
    # model = Prop_Interaction(config)
    # out = model(prop_feature)
    # print(out.shape)

    import torch.optim as optim
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, verbose=False)

    for i in range(10):

        scheduler.step(1.)
        print('Epoch ', i, optimizer.param_groups[0]['lr'])
        # print()