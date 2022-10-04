import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import numpy as np
import math

from feature_extraction import QueryEncoder, VisualEncoder, CQAttention
from feature_enhancement.boundary_content import TemporalDifference
from proposal_construction.segment_gen import SparseBoundaryCat, SparseMaxPool, DenseMaxPool
from utils import sequence2mask, PropPositionalEncoding
from loss import MainLoss as mainloss
from proposal_refinement import NaivePredictor, Adaptive_Prop_Interaction, Aaptive_Proposal_Sampling


class BAN(nn.Module):
    def __init__(self, vocab_size, cfg, pre_train_emb=None, device='cpu'):
        super(BAN, self).__init__()
        self.device = device
        self.max_video_seq_len = cfg.model.video_seq_len
        self.topk = cfg.model.topk
        self.neighbor = cfg.model.neighbor
        self.negative = cfg.model.negative
        self.prop_num = cfg.model.prop_num

        self.visual_encoder = VisualEncoder(cfg.model.visual_embed_dim, cfg.model.hidden_dim, cfg.model.lstm_layer)
        self.query_encoder = QueryEncoder(vocab_size, cfg.model.hidden_dim, embed_dim=cfg.model.query_embed_dim,
                                          num_layers=cfg.model.lstm_layer, pre_train_weights=pre_train_emb)
        self.cross_encoder = VisualEncoder(4 * cfg.model.fuse_dim, cfg.model.hidden_dim,
                                           cfg.model.lstm_layer)
        self.cqa_att = CQAttention(cfg.model.fuse_dim)
        self.boundary_aware = TemporalDifference(cfg, in_dim=cfg.model.fuse_dim, layer_num=2)
        self.boundary_aggregation = SparseBoundaryCat(cfg.model.pooling_counts, self.max_video_seq_len, device)
        if cfg.model.sparse_sample:
            self.content_aggregation = SparseMaxPool(cfg.model.pooling_counts, self.max_video_seq_len, device)
        else:
            self.content_aggregation = DenseMaxPool(self.max_video_seq_len, device)
        self.map2d_proj = nn.Sequential(
            nn.Linear(3 * cfg.model.fuse_dim, cfg.model.fuse_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False)
        )
        self.prop_sampler = Aaptive_Proposal_Sampling(self.topk, self.neighbor, self.negative, 0.7)
        self.predictor = NaivePredictor(cfg.model.fuse_dim, cfg.model.fuse_dim, intermediate=True)
        self.predictor2 = NaivePredictor(cfg.model.fuse_dim, cfg.model.fuse_dim, intermediate=True)
        self.predictor_offset = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.fuse_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(cfg.model.fuse_dim, 2)
        )
        self.prop_pe = PropPositionalEncoding(cfg.model.fuse_dim, cfg.model.hidden_dim)
        self.fc_fuse = nn.Linear(6 * cfg.model.hidden_dim, cfg.model.fuse_dim)
        self.contrast_encoder = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.contrast_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model.contrast_dim, cfg.model.contrast_dim)
        )
        self.contrast_encoder_t = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.contrast_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model.contrast_dim, cfg.model.contrast_dim)
        )
        self.prop_interact = Adaptive_Prop_Interaction(cfg)
        self.loss = mainloss(cfg.loss.min_iou, cfg.loss.max_iou)

    def forward(self, data):
        data_visual, data_text, video_seq_len, text_seq_len, offset_gt = \
            data['v_feature'], data['q_feature'], data['v_len'], data['q_len'],  data['start_end_offset']
        # feature encoder
        video_feature, clip_feature = self.visual_encoder(data_visual, video_seq_len, self.max_video_seq_len)
        sentence_feature, word_feature = self.query_encoder(data_text, text_seq_len)
        mask_word = sequence2mask(text_seq_len)
        cat_feature = self.cqa_att(clip_feature, word_feature, mask_word)
        _, fuse_feature = self.cross_encoder(cat_feature, video_seq_len, self.max_video_seq_len)
        # boundary prediction
        out = self.boundary_aware(fuse_feature)
        hidden_b, hidden_c = out['feature']
        td = out['td']  # (bs, seq)
        # proposal generation
        map2d_s_e, _ = self.boundary_aggregation(hidden_b.permute(0, 2, 1), hidden_b.permute(0, 2, 1))
        map2d_c, map2d_mask = self.content_aggregation(fuse_feature.permute(0, 2, 1))
        map2d_c = map2d_c.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        map2d_s_e = map2d_s_e.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        map2d_sec = torch.cat([map2d_s_e, map2d_c], dim=-1)
        map2d = self.map2d_proj(map2d_sec)
        # matching prediction
        tmap = self.predictor(map2d)
        # content feature for contrastive learning
        map2d_proj = self.contrast_encoder(map2d_c)
        sen_proj = self.contrast_encoder_t(sentence_feature)
        # fuse_feature = self.frame_pos(fuse_feature)
        B, N, D = hidden_c.size()
        score_pred = tmap.sigmoid() * map2d_mask
        score_pred = score_pred.clone().detach()
        prop_feature, pred_s_e, offset_gt, pred_score = \
            self.prop_sampler(score_pred, map2d_mask, map2d, offset_gt, tmap)

        prop_feature = self.prop_pe(prop_feature.view(-1, D), pred_s_e.view(-1, 2))
        prop_num = self.prop_num

        prop_feature = prop_feature.view(B, prop_num, D)
        pred_s_e = pred_s_e.view(B, prop_num, 2)
        offset_gt = offset_gt.view(B, prop_num, 2)
        pred_score = pred_score.view(B, prop_num)
        # proposal interaction and matching score prediction
        # prop_feature = self.prop_interact(prop_feature, mask_prop)
        prop_feature = self.prop_interact(prop_feature)
        pred = self.predictor2(prop_feature)
        offset = self.predictor_offset(prop_feature)

        out = {'tmap': tmap,
               'map2d_mask': map2d_mask,
               'map2d_proj': map2d_proj,
               'sen_proj': sen_proj,
               'coarse_pred': pred_s_e,
               'coarse_pred_round': pred_s_e,
               'final_pred': pred,
               'offset': offset,
               'offset_gt': offset_gt,
               }

        loss = self.loss(out, data, td)
        return out, loss


if __name__ == "__main__":
    # x = torch.randn(size=(2, 10, 48))  # B, D, N
    #
    # pooling_counts = [11, 6, 6]
    # N = 48
    # model = SparseMaxPool(pooling_counts, N)
    # scores2d, mask2d = model(x)
    # scores2d = torch.randn(size=(1, 48, 48))
    # ious2d = torch.randn(size=(1, 48, 48))
    from charades_config import config
    from co_trm.predictor import Predictor2
    from segment_gen import AdaptivePropGen

    B, N, D = (2, 48, 512)
    x = torch.randn(size=(B, N, D))  # B, N, D
    model = AdaptivePropGen(config)
    prop, coarse_pred, s_e_round = model(x)
    model = Predictor2(config)
    pred = model(prop, s_e_round)
    x = 1

    pass
