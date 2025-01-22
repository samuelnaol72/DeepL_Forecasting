import torch
import numpy as np
from utils.benchmark import benchmark, metric_with_times


def test_model(config, trained_model, val_loader):
    """
    Evaluates a trained model on the validation set, computes metrics, and returns results.

    Parameters:
        config (dict): Configuration dictionary containing evaluation parameters.
        trained_model (torch.nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        dict: Dictionary containing:
            - 'true_values' (numpy.ndarray): True target values.
            - 'predicted_values' (numpy.ndarray): Predicted target values.
            - 'metrics' (dict): Evaluation metrics (e.g., RMSE, MAE, precision).
            - 'avg_inference_time' (float): Average inference time per batch.
    """
    # Set device
    device = torch.device(
        config["device"]["type"] if torch.cuda.is_available() else "cpu"
    )

    # Ensure the model is on the correct device and set to evaluation mode
    trained_model.to(device)
    trained_model.eval()

    # Initialize variables for evaluation
    y_true = []
    y_pred = []
    inference_times = []

    print("Evaluating the model on the validation dataset...")

    # Evaluate on the validation loader
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Measure inference time
            inf_start = time.time()
            preds = trained_model(X_batch)
            inf_end = time.time()

            inference_times.append(inf_end - inf_start)

            # Store predictions and true values
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute inference time statistics
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(val_loader)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute regression metrics
    metrics = benchmark(y_true, y_pred)

    # Compute additional metrics with timing
    timing_metrics = metric_with_times(
        y_test=y_true,
        y_pred=y_pred,
        train_time=config.get("training_time", 0),  # Default 0 if not logged
        inf_time=avg_inference_time,
    )

    # Print results
    print("Benchmark Metrics:")
    print(metrics)

    print("Timing Metrics:")
    print(timing_metrics)

    # Combine metrics
    combined_metrics = {
        **metrics.to_dict(orient="records")[0],
        **timing_metrics.to_dict(orient="records")[0],
    }

    # Return results
    return {
        "true_values": y_true,
        "predicted_values": y_pred,
        "metrics": combined_metrics,
        "avg_inference_time": avg_inference_time,
    }
