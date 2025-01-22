import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Single block of the N-BEATS model.

        Args:
            input_size: Number of input features (lag or look-back window size).
            hidden_size: Number of neurons in each hidden layer.
            output_size: Number of target values (forecast horizon).
        """
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(
            hidden_size, input_size + output_size
        )  # Backcast + Forecast

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through a single N-Beats block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Concatenation of backcast and forecast.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        backcast_forecast = self.fc4(x)
        return backcast_forecast


class NBeatsModel(nn.Module):
    def __init__(self, config):
        """
        N-BEATS model with multiple blocks.

        Args:
            config (dict): Configuration dictionary containing model parameters:
                - 'input_size': Number of input features (lag or look-back window size).
                - 'hidden_size': Number of neurons in each hidden layer.
                - 'output_size': Number of target values (forecast horizon).
                - 'num_blocks': Number of stacked blocks.
        """
        super(NBeatsModel, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_blocks = config.get("num_blocks", 3)

        # Create multiple blocks
        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(self.input_size, self.hidden_size, self.output_size)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, x):
        """
        Forward pass through the N-BEATS model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Forecast output of shape (batch_size, output_size).
        """
        backcast = x
        forecast = torch.zeros(x.size(0), self.output_size).to(x.device)

        for block in self.blocks:
            backcast_forecast = block(backcast)
            block_backcast = backcast_forecast[:, : self.input_size]  # Extract backcast
            block_forecast = backcast_forecast[:, self.input_size :]  # Extract forecast

            backcast = backcast - block_backcast  # Update residual backcast
            forecast = forecast + block_forecast  # Aggregate forecast

        return forecast
