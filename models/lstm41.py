import torch
from torch import nn
import torch.nn.functional as F


class LSTM41(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layer):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layer = num_stacked_layer

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layer, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        self.h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)
        self.c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])

        return out
