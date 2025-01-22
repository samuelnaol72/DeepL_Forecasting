import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, config):
        """
        Initializes the LSTM model based on the given configuration.

        Parameters:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size' (int): Number of input features.
                - 'hidden_size' (int): Number of hidden units in the LSTM.
                - 'num_layers' (int): Number of stacked LSTM layers.
                - 'output_size' (int): Number of output features (lookhour).
        """
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.output_size = config.get("output_size", 1)  # Default to 1 if not provided

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Extract the last hidden state
        out = out[:, -1, :]

        # Pass through the fully connected layer
        out = self.fc(out)

        return out
