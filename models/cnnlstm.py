import torch
from torch import nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layer, cnn_output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layer = num_stacked_layer

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_output_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=cnn_output_size,
            out_channels=cnn_output_size,
            kernel_size=3,
            padding=1,
        )

        self.lstm_input_size = cnn_output_size
        self.lstm = nn.LSTM(
            self.lstm_input_size, hidden_size, num_stacked_layer, batch_first=True
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

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)

        self.h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)
        self.c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size)

        out, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])
        return out
