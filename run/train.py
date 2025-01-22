import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.log import log_metrics
from utils.saver import save_model
from models.cnnlstm import CNNLSTM
from models.gru import GRU
from models.lstm import LSTM
from models.lstm41 import LSTM41
from models.lstmreset import LSTMRESET
from models.nbeats import NBeatsModel


def train_model(config, train_loader, val_loader):
    """
    Trains a model using the provided data loaders, saves the trained model,
    and returns the model along with the training loss trend.

    Parameters:
        config (dict): Configuration dictionary containing model and training parameters.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - loss_trend (list): List of training loss values over epochs.
    """
    # Set device
    device = torch.device(
        config["device"]["type"] if torch.cuda.is_available() else "cpu"
    )

    # Model mapping for dynamic selection
    model_mapping = {
        "cnn_lstm": CNNLSTM,
        "gru": GRU,
        "lstm": LSTM,
        "lstm41": LSTM41,
        "lstmreset": LSTMRESET,
        "nbeats": NBeatsModel,
    }

    # Validate model name
    model_name = config["model"]["name"]
    if model_name not in model_mapping:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Available options are: {list(model_mapping.keys())}"
        )

    # Initialize the model
    model_class = model_mapping[model_name]
    model = model_class(config["model"]).to(device)

    # Initialize optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = MSELoss()

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("scheduler_gamma", 0.1),
        patience=config["training"].get("scheduler_patience", 5),
        verbose=True,
    )

    # Store training loss trend
    loss_trend = []

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        loss_trend.append(avg_train_loss)

        # Validation
        val_loss = log_metrics(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']}: "
            f"Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, "
            f"Learning Rate = {optimizer.param_groups[0]['lr']:.6f}"
        )

    # Save the trained model
    save_model(model, config["model"]["save_path"])
    print(f"Model saved at {config['model']['save_path']}")

    return model, loss_trend
