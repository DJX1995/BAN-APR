import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pad_sequence  # used in pad_collate
import numpy as np
import logging
import time
import math


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
