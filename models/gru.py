import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, config):
        """
        Initializes the GRU model based on the given configuration.

        Parameters:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size' (int): Number of input features.
                - 'hidden_size' (int): Number of hidden units in the GRU.
                - 'num_layers' (int): Number of stacked GRU layers.
                - 'output_size' (int): Number of output features (lookahead steps).
        """
        super(GRU, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]

        # GRU layer
        self.gru = nn.GRU(
            input_size=config["input_size"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, config["output_size"])

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.gru(x, h0)

        # Decode the last hidden state
        out = self.fc(out[:, -1, :])
        return out
