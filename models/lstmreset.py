import torch
from torch import nn


class LSTMRESET(nn.Module):
    def __init__(self, config):
        """
        Initializes the LSTMRESET model using the configuration dictionary.

        Parameters:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size' (int): Number of input features.
                - 'hidden_size' (int): Number of hidden units in the LSTM.
                - 'num_layers' (int): Number of stacked LSTM layers.
                - 'output_size' (int): Number of target output features.
                - 'dropout' (float): Dropout rate for the LSTM.
        """
        super(LSTMRESET, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_stacked_layer = config["num_layers"]
        self.output_size = config["output_size"]

        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            config["input_size"],
            self.hidden_size,
            self.num_stacked_layer,
            batch_first=True,
            dropout=config.get("dropout", 0.1),
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Initialize hidden and cell states
        self.h0 = None
        self.c0 = None
        self.current_batch_size = None

    def reset_hidden_state(self, batch_size, device):
        """
        Resets the hidden state and cell state.

        Parameters:
            batch_size (int): Batch size for the hidden and cell states.
            device (torch.device): Device to allocate the hidden states.
        """
        self.h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            device
        )
        self.c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size).to(
            device
        )
        self.current_batch_size = batch_size

    def forward(self, x):
        """
        Forward pass through the LSTMRESET model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)

        # Reset hidden states if batch size changes or states are uninitialized
        if self.h0 is None or self.c0 is None or batch_size != self.current_batch_size:
            self.reset_hidden_state(batch_size, x.device)

        # Detach hidden states to prevent gradient backpropagation through time
        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()

        # Forward pass through LSTM
        out, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))

        # Pass through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
