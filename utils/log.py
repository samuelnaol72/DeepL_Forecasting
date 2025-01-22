import torch


def log_metrics(model, loader, criterion, device):
    """
    Evaluates the model on a given DataLoader and logs validation metrics.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the validation/test set.
        criterion (torch.nn.Module): Loss function to evaluate the model.
        device (torch.device): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        float: Average loss over the DataLoader.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Compute loss
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    # Compute average loss
    avg_loss = total_loss / len(loader)

    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
