import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pad_sequence  # used in pad_collate
import skimage.measure
import numpy as np
import logging
import time
import random
import math

seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # if using cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PropPositionalEncoding(nn.Module):
    def __init__(self, dim_in=512, dim_emb=256, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, dim_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_emb, 2).float() * (-math.log(10000.0) / dim_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(dim_in + 2*dim_emb, dim_in)

    def forward(self, x, prop_s_e):
        # x: (B, N, hidden) or (B*N, hidden)
        if x.dim() == 3:
            B, N, D = x.size()
            all_num = B * N
            x = x.view(-1, D)  # (prop_num, D)
        else:
            all_num, D = x.size()
        s, e = prop_s_e[:, 0], prop_s_e[:, 1]
        pe_table = self.pe.repeat(all_num, 1, 1)
        pos_s = pe_table[torch.arange(all_num), s, :]
        pos_e = pe_table[torch.arange(all_num), e-1, :]
        # pos_s = pos_s.view(N, -1)
        # pos_e = pos_e.view(N, -1)
        x = torch.cat([x, pos_s, pos_e], dim=-1)
        x = self.fc(x)
        if x.dim() == 3:
            x = x.view(B, N, D)
        return x


def evaluation(pred, target):
    '''
    used for model evaluation, tensor operation, return IOU between pred and target
    pred: (batch,2)
    target: (batch,2)
    '''
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]

    target_left = target[:, 0]
    target_right = target[:, 1]

    intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
    intersect[intersect < 0] = 0
    target_area = target_right - target_left
    pred_area = pred_right - pred_left
    pred_area[pred_area < 0] = 0
    union = target_area + pred_area - intersect

    iou = (intersect + 1e-8) / (union + 1e-8)

    assert iou.numel() != 0
    return iou


def pool_feature(max_video_seq_len, feature):
    '''
    all are numpy array
    :param max_video_seq_len: max_seq_len
    :param feature: in size of (seq,hidden)
    :return pool_feature: in size of (max_video_seq_len,hidden)
    '''
    kernel_size = feature.shape[0] // max_video_seq_len
    num_kernel_plus = feature.shape[0] % max_video_seq_len
    num_kernel = max_video_seq_len - num_kernel_plus
    #     print(f'{num_kernel} blocks in size ({kernel_size},{feature.shape[1]})\
    # and {num_kernel_plus} blocks in size ({kernel_size+1},{feature.shape[1]})')
    try:
        head = skimage.measure.block_reduce(feature, (kernel_size, 1), np.max)[:num_kernel]
        tail = skimage.measure.block_reduce(feature[::-1, :], (kernel_size + 1, 1), np.max)[:num_kernel_plus][::-1, :]
    except:
        print(f'feature size is {feature.shape}, max seq len is {max_video_seq_len}')
        return None
    pool_feature = np.concatenate([head, tail], axis=0)
    return pool_feature


def bool2index(bool_label):
    # bool_label has 1/0 value of shape (seq_len,), used for contrastive loss
    mask = bool_label.ge(0.5)
    # temp = mask.new_zeros(mask.shape)
    try:
        index = torch.masked_select(torch.arange(0, len(bool_label)).to(mask.device), mask)
    except:
        print(bool_label)
    return index


def contrast_selection(contrast_label, feature):
    '''
    # used for contrastive loss
    :param contrast_label: contain positive and negative index (1 and 0 values)
    :param feature: (seq,hidden)
    :return: positive features and negative features
    '''

    pos_index = bool2index(contrast_label)
    neg_index = bool2index(1 - contrast_label)
    return torch.index_select(feature, dim=0, index=pos_index), torch.index_select(feature, dim=0, index=neg_index)


def get_attquery_feature(feature_t, feature_v, query_length, mean_pool=False,
                         uniform_att=False, return_raw=False, norm=False):
    '''
    input
    -----
    text_hidden: (batch, seq_len, hidden)
    text_hidden: (batch, video_len, hidden)
    query_length: (batch,) indicates seq_len of each sentence

    return
    ------
    query_feature: (batch, video_len, hidden)
    '''
    if norm:
        text_hidden = feature_t / (torch.linalg.norm(feature_t, dim=-1, keepdim=True) + 1e-8)
        visual_hidden = feature_v / (torch.linalg.norm(feature_v, dim=-1, keepdim=True) + 1e-8)
    else:
        text_hidden = feature_t
        visual_hidden = feature_v
    max_visualseq_len = visual_hidden.size(1)
    att_map = torch.matmul(visual_hidden, text_hidden.permute(0, 2, 1))
    if uniform_att:
        att_map = torch.ones_like(att_map)
    batch_size, num_visual, num_loc = att_map.size()
    tmp1 = att_map.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=att_map.dtype).unsqueeze(0)
    tmp2 = query_length.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)

    mask = torch.ge(tmp1, tmp2)
    mask = mask.unsqueeze(1).repeat(1, max_visualseq_len, 1)
    att_map = att_map.masked_fill(mask, -1e30)
    raw_att = torch.exp(att_map)
    att_map = att_map.softmax(dim=-1)

    # attended query feature should be in shape of (batch, max_visualseq_len, hidden), whcch is (3,8,10) here
    if mean_pool:
        query_length = query_length.unsqueeze(-1).unsqueeze(-1)
    else:
        query_length = torch.ones_like(query_length).unsqueeze(-1).unsqueeze(-1)
    # query_feature = torch.matmul(att_map, text_hidden) / query_length
    query_feature = torch.matmul(att_map, feature_t) / query_length
    if return_raw:
        return query_feature, raw_att
    return query_feature, att_map


# def get_attquery_feature(feature_t, feature_v, query_length, mean_pool=False,
#                          uniform_att=False, return_raw=False):
#     '''
#     input
#     -----
#     text_hidden: (batch, seq_len, hidden)
#     text_hidden: (batch, video_len, hidden)
#     query_length: (batch,) indicates seq_len of each sentence

#     return
#     ------
#     query_feature: (batch, video_len, hidden)
#     '''
#     text_hidden = feature_t / (torch.linalg.norm(feature_t, dim=-1, keepdim=True) + 1e-8)
#     visual_hidden = feature_v / (torch.linalg.norm(feature_v, dim=-1, keepdim=True) + 1e-8)
#     max_visualseq_len = visual_hidden.size(1)
#     att_map = torch.matmul(visual_hidden, text_hidden.permute(0, 2, 1))
#     if uniform_att:
#         att_map = torch.ones_like(att_map)
#     batch_size, num_visual, num_loc = att_map.size()
#     tmp1 = att_map.new_zeros(num_loc)
#     tmp1[:num_loc] = torch.arange(
#         0, num_loc, dtype=att_map.dtype).unsqueeze(0)
#     tmp2 = query_length.type(tmp1.type())
#     tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)

#     mask = torch.ge(tmp1, tmp2)
#     mask = mask.unsqueeze(1).repeat(1, max_visualseq_len, 1)
#     att_map = att_map.masked_fill(mask, -1e30)
#     raw_att = torch.exp(att_map)
#     att_map = att_map.softmax(dim=-1)

#     # attended query feature should be in shape of (batch, max_visualseq_len, hidden), whcch is (3,8,10) here
#     if mean_pool:
#         query_length = query_length.unsqueeze(-1).unsqueeze(-1)
#     else:
#         query_length = torch.ones_like(query_length).unsqueeze(-1).unsqueeze(-1)
#     # query_feature = torch.matmul(att_map, text_hidden) / query_length
#     query_feature = torch.matmul(att_map, feature_t) / query_length
#     if return_raw:
#         return query_feature, raw_att
#     return query_feature, att_map


def iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def sequence2mask(seq_len, maxlen=None):
    # seq_len: (batch, 1) or (batch, )
    seq_len = seq_len.squeeze()
    seq_len = seq_len.unsqueeze(-1)
    batch_size = len(seq_len)
    if maxlen is None:
        maxlen = seq_len.max()
    tmp1 = torch.arange(0, maxlen, device=seq_len.device).unsqueeze(0)
    tmp2 = seq_len.type(tmp1.type())
    tmp2 = tmp2.expand(batch_size, maxlen)
    mask = torch.ge(tmp1, tmp2)
    return ~mask


def Prepare_logger(log_name, print_console=True):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if print_console:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = log_name + date + '.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def conv1d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True, padding=None):
    if not padding:
        padding = dilation * (kernel_size - 1) // 2
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    )
    nn.init.kaiming_uniform_(conv.weight, a=1)
    return conv


def make_conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=None,
              use_bn=True, use_relu=True, use_dropout=False):
    bias = False if use_bn else True
    conv = conv1d(in_channels, out_channels, kernel_size, stride, dilation, bias, padding)
    module = [conv, ]
    if use_bn:
        module.append(nn.BatchNorm1d(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if use_dropout:
        module.append(nn.Dropout(p=0.5))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv