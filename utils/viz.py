import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(metrics):
    """
    Plots key metrics (e.g., RMSE, MAE, Precision, Recall).

    Parameters:
        metrics (dict): Dictionary containing evaluation metrics (e.g., RMSE, MAE, precision, recall).
    """
    # Convert metrics to DataFrame for plotting
    metrics_df = pd.DataFrame([metrics])

    # Plot RMSE and MAE
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["RMSE", "MAE"], [metrics["RMSE"], metrics["MAE"]], color=["blue", "orange"]
    )
    plt.title("Model Performance Metrics (RMSE and MAE)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    # Plot Precision and Recall if they exist in the metrics
    if "Freezing Precision" in metrics and "Freezing Recall" in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(
            ["Precision", "Recall"],
            [metrics["Freezing Precision"], metrics["Freezing Recall"]],
            color=["green", "red"],
        )
        plt.title("Freezing Prediction Metrics (Precision and Recall)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()


def plot_predictions(actual, predicted):
    """
    Plots predictions vs. actual values.

    Parameters:
        actual (numpy.ndarray): Actual target values.
        predicted (numpy.ndarray): Predicted target values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual.flatten(), label="True Values")
    plt.plot(predicted.flatten(), label="Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Predicted vs. True Values")
    plt.grid(True)
    plt.show()


def plot_training_loss(loss_values):
    """
    Plots training loss over epochs and saves loss values to a file.

    Parameters:
        loss_values (list): List of loss values for each epoch.
    """
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()

    # Save training loss to a file
    file_name = "training_loss.txt"
    with open(file_name, "w") as file:
        for item in loss_values:
            file.write(f"{item}\n")

    print(f"Loss values saved to {file_name}")


def visualize_results(inf_result, loss_values):
    """
    Visualizes metrics, predictions vs. actuals, and training loss using the results from the `test_model` function.

    Parameters:
        inf_result (dict): Result dictionary from `test_model`, containing:
            - 'true_values': Actual target values.
            - 'predicted_values': Predicted target values.
            - 'metrics': Evaluation metrics as a dictionary.
            - 'avg_inference_time': Average inference time per batch.
        loss_values (list): List of training loss values over epochs.
    """
    # Extract data from inf_result
    actual = inf_result["true_values"]
    predicted = inf_result["predicted_values"]
    metrics = inf_result["metrics"]
    avg_inference_time = inf_result["avg_inference_time"]

    # Display timing information
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

    # Plot metrics
    plot_metrics(metrics)

    # Plot predictions vs. actuals
    plot_predictions(actual, predicted)

    # Plot training loss
    plot_training_loss(loss_values)
