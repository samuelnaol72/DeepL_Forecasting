import torch
from torch import nn
import torch.nn.functional as F


class LSTMRESET(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layer):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layer = num_stacked_layer

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layer, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, 8)

        self.h0 = None
        self.c0 = None
        self.current_batch_size = None

    def reset_hidden_state(self, batch_size):

        self.h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)
        self.c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)
        self.current_batch_size = batch_size

    def forward(self, x):
        batch_size = x.size(0)

        if self.h0 is None or self.c0 is None or batch_size != self.current_batch_size:
            self.reset_hidden_state(batch_size)

        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()

        out, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])
        return out
