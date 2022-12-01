import torch
from torch import nn


class NaivePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, intermediate=True, drop_rate=0.1):
        super().__init__()
        if intermediate:
            self.pred = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop_rate, inplace=False),
                    nn.Linear(hidden_size, 1)
                )
        else:
            self.pred = nn.Linear(input_size, 1)

    def forward(self, x):
        tmap_logit = self.pred(x)
        return tmap_logit.squeeze(-1)




















