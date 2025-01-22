from torch.utils.data import DataLoader
from data_provider.data_loader import TimeSeriesDataset
from data_provider.data_preprocess import preprocessing
import logging
import time
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Timing and logging decorator for monitoring execution
def timing_and_logging(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(
            f"Starting function: {func.__name__} with args={args[1:]}, kwargs={kwargs}"
        )
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f"Completed {func.__name__} in {end_time - start_time:.2f} seconds"
        )
        return result

    return wrapper


@timing_and_logging
def create_loaders(df, cfg):
    """
    Prepares PyTorch DataLoaders for training and validation using the given configuration.

    Parameters:
        df (pd.DataFrame): Input dataframe containing raw data.
        cfg (dict): Configuration dictionary containing hyperparameters, such as:
            - 'lags' (int): Number of lagged features to include.
            - 'horizon' (int): Number of future steps to predict.
            - 'ids' (list): List of unique IDs for preprocessing.
            - 'batch_size' (int): Batch size for DataLoaders.
            - 'shuffle' (bool): Whether to shuffle the DataLoader during training.
            - 'model' (str): Type of model to use (e.g., 'lstm', 'gru').
            - 'device' (str): Device to run the tensors on ('cpu' or 'cuda').
            - 'channels' (int): Number of channels in the target output.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
    """
    # Extract configuration parameters with defaults
    lags = cfg.get("lags", 24)
    horizon = cfg.get("horizon", 8)
    ids = cfg.get("ids", [])
    batch_size = cfg.get("batch_size", 32)
    shuffle = cfg.get("shuffle", True)
    model = cfg.get("model", "lstm")
    device = cfg.get("device", "cpu")
    channels = cfg.get("channels", 1)

    logging.info(f"Configuration: {cfg}")

    # Preprocess the data to generate train and validation splits
    X_train, y_train, X_val, y_val = preprocessing(df, lags, horizon, ids)

    # Reshape data for LSTM/GRU models if required
    if model in ["lstm", "gru", "lstm41", "lstmreset", "cnnlstm"]:
        X_train = X_train.reshape((-1, 31, channels))
        X_val = X_val.reshape((-1, 31, channels))
        y_train = y_train.reshape((-1, horizon))
        y_val = y_val.reshape((-1, horizon))

    # Convert to PyTorch tensors and move to the specified device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    logging.info(f"Train set: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Validation set: X_val={X_val.shape}, y_val={y_val.shape}")

    # Create datasets for training and validation
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
