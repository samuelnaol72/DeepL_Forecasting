import torch
from torch import nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, config):
        """
        Initializes the CNNLSTM model using the configuration dictionary.

        Parameters:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size' (int): Number of input features.
                - 'hidden_size' (int): Number of hidden units in the LSTM.
                - 'num_layers' (int): Number of stacked LSTM layers.
                - 'cnn_output_size' (int): Number of output channels for CNN layers.
                - 'output_size' (int): Number of target output features.
        """
        super(CNNLSTM, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_stacked_layer = config["num_layers"]

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=config["input_size"],
            out_channels=config["cnn_output_size"],
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config["cnn_output_size"],
            out_channels=config["cnn_output_size"],
            kernel_size=3,
            padding=1,
        )

        # LSTM layer
        self.lstm_input_size = config["cnn_output_size"]
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.hidden_size,
            self.num_stacked_layer,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, config["output_size"])

        # Hidden states
        self.h0 = None
        self.c0 = None

    def reset_hidden_state(self, batch_size, device):
        """
        Resets the hidden states (h0, c0) for the LSTM layer.

        Parameters:
            batch_size (int): Size of the current batch.
            device (torch.device): Device to initialize the hidden states on.
        """
        self.h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            device
        )
        self.c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            device
        )

    def forward(self, x):
        """
        Forward pass through the CNNLSTM model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)

        # CNN layers
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, cnn_output_size)

        # Reset hidden state for LSTM
        self.reset_hidden_state(batch_size, x.device)

        # LSTM layer
        out, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))

        # Fully connected layer
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out
