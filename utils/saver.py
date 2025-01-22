import torch


def save_model(model, save_path):
    """Saves the model to a file."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
