import torch
from torch import nn


class LSTM41(nn.Module):
    def __init__(self, config):
        """
        Initializes the LSTM41 model using the configuration dictionary.

        Parameters:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size' (int): Number of input features.
                - 'hidden_size' (int): Number of hidden units in the LSTM.
                - 'num_layers' (int): Number of stacked LSTM layers.
                - 'output_size' (int): Number of target output features.
        """
        super(LSTM41, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_stacked_layer = config["num_layers"]

        # LSTM layer
        self.lstm = nn.LSTM(
            config["input_size"],
            self.hidden_size,
            self.num_stacked_layer,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, config["output_size"])

    def forward(self, x):
        """
        Forward pass through the LSTM41 model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            x.device
        )

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layer using the last hidden state
        out = self.fc(out[:, -1, :])

        return out
