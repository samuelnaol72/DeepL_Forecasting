import torch
from torch.optim import Adam
from torch.nn import MSELoss
from utils.log import log_metrics
from utils.saver import save_model


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

    # Initialize model, optimizer, and loss function
    model_class = getattr(
        __import__("models", fromlist=[config["model"]["name"]]),
        config["model"]["name"],
    )
    model = model_class(config["model"]).to(device)
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = MSELoss()

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

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']}: "
            f"Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    # Save the trained model
    save_model(model, config["model"]["save_path"])
    print(f"Model saved at {config['model']['save_path']}")

    return model, loss_trend
