import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def temporaldifference(feature):
    # (bs, seq, hidden)
    feature_rpad = F.pad(feature.permute(0, 2, 1), (0, 1))  # (bs, hidden, seq + 1)
    feature_lpad = F.pad(feature.permute(0, 2, 1), (1, 0))  # (bs, hidden, seq + 1)
    feature_rpad[:, :, -1] = feature.permute(0, 2, 1)[:, :, -1]
    feature_lpad[:, :, 0] = feature.permute(0, 2, 1)[:, :, 0]
    td_1 = feature_rpad[:, :, 1:] - feature.permute(0, 2, 1)  # (bs, hidden, seq)
    td_2 = feature_lpad[:, :, :-1] - feature.permute(0, 2, 1)  # (bs, hidden, seq)
    td = td_1.square() + td_2.square()
    td = td.permute(0, 2, 1)  # (bs, seq, hidden)
    return td


class TemporalDifference(nn.Module):
    def __init__(self, config, in_dim=None, model_type='lstm', layer_num=1):
        super().__init__()
        self.split_dim = config.model.fuse_dim
        if in_dim == None:
            in_dim = self.split_dim
        self.model_type = model_type
        if model_type == 'lstm':
            self.feature_transform_b = nn.LSTM(in_dim, self.split_dim, layer_num,
                                               batch_first=True, bidirectional=True)
            self.feature_transform_c = nn.LSTM(in_dim, self.split_dim, layer_num,
                                               batch_first=True, bidirectional=True)
            self.feature_proj_b = nn.Sequential(
                nn.Linear(2 * self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.drop_rate, inplace=False)
            )
            self.feature_proj_c = nn.Sequential(
                nn.Linear(2 * self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.drop_rate, inplace=False)
            )
        elif model_type == 'cnn':
            self.feature_transform_b = torch.nn.Sequential(
                nn.Conv1d(self.split_dim, self.split_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.split_dim),
                nn.ReLU()
            )
            self.feature_transform_c = torch.nn.Sequential(
                nn.Conv1d(self.split_dim, self.split_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.split_dim),
                nn.ReLU()
            )
            self.feature_proj_b = nn.Sequential(
                nn.Linear(self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.drop_rate, inplace=False)
            )
            self.feature_proj_c = nn.Sequential(
                nn.Linear(self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.drop_rate, inplace=False)
            )
        else:
            raise NotImplementedError("sequence model in TD not implemented!")

    def forward(self, visual_input):
        # (B, T, D)
        if self.model_type == 'lstm':
            hidden_b, _ = self.feature_transform_b(visual_input)
            hidden_c, _ = self.feature_transform_c(visual_input)
        elif self.model_type == 'cnn':
            hidden_b = self.feature_transform_b(visual_input.permute(0, 2, 1)).permute(0, 2, 1)
            hidden_c = self.feature_transform_c(visual_input.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_b = self.feature_proj_b(hidden_b)
        hidden_c = self.feature_proj_c(hidden_c)
        td = temporaldifference(hidden_b)  # (bs, seq, hidden)
        td = td.sum(dim=-1)
        return {'feature': [hidden_b, hidden_c],
                'td': td}


