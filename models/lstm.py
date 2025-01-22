import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layer, lookhour):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layer = num_stacked_layer

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layer, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, lookhour)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        # Extract only the last hidden state for a given sequence
        # Is this efficient??
        out = out[:, -1, :]

        out = self.fc(out)

        return out
