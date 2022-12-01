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


def adaptive_graph_feature(x):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    x = x.transpose(2, 1).contiguous()
    tmp1 = x.unsqueeze(1).repeat(1, num_points, 1, 1)
    tmp2 = x.unsqueeze(2).repeat(1, 1, num_points, 1)
    tmp3 = tmp1 - tmp2
    feature = torch.cat((tmp3, tmp2), dim=3).permute(0, 3, 1, 2)
    return feature


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

