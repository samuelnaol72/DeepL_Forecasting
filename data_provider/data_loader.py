from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset class for time series data, which can be used with
    PyTorch's DataLoader to efficiently batch and iterate through the data.

    Attributes:
        X (pd.DataFrame or np.ndarray): Input features for the dataset.
        y (pd.DataFrame or np.ndarray): Target values for the dataset.
    """

    def __init__(self, x_data, y_data):
        """
        Initializes the TimeSeriesDataset with input features and targets.

        Parameters:
            x_data (pd.DataFrame or np.ndarray): Input features of the dataset.
            y_data (pd.DataFrame or np.ndarray): Target values of the dataset.
        """
        self.X = x_data  # Store input features
        self.y = y_data  # Store target values

    def __len__(self):
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        Retrieves a single sample from the dataset.

        Parameters:
            i (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input features and target values
                   for the i-th sample.
        """
        return self.X[i], self.y[i]
