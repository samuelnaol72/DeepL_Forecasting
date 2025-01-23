import pandas as pd
import matplotlib.pyplot as plt


def plot_hour_metrics(metric_results):
    """
    Plots RMSE, MAE, Precision, and Recall across prediction hours.

    Parameters:
        metric_results (dict): Dictionary containing hour-wise metric results.
    """
    metric_df = pd.DataFrame(metric_results)

    # Plot RMSE and MAE
    plt.figure(figsize=(10, 6))
    plt.plot(metric_df["Hour"], metric_df["RMSE"], marker="o", label="RMSE")
    plt.plot(metric_df["Hour"], metric_df["MAE"], marker="s", label="MAE")
    plt.title("RMSE and MAE Across Prediction Hours")
    plt.xlabel("Prediction Hour")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Precision and Recall
    if "Freezing Precision" in metric_df and "Freezing Recall" in metric_df:
        plt.figure(figsize=(10, 6))
        plt.plot(
            metric_df["Hour"],
            metric_df["Freezing Prediction Precision"],
            marker="o",
            label="Precision",
            color="blue",
        )
        plt.plot(
            metric_df["Hour"],
            metric_df["Freezing Prediction Recall"],
            marker="s",
            label="Recall",
            color="green",
        )
        plt.title("Freezing Prediction Precision and Recall Across Prediction Hours")
        plt.xlabel("Prediction Hour")
        plt.ylabel("Metric Value")
        plt.legend()
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
    Visualizes hour-wise metrics, predictions vs. actuals, and training loss.

    Parameters:
        inf_result (dict): Result dictionary from `test_model`, containing:
            - 'true_values': Actual target values.
            - 'predicted_values': Predicted target values.
            - 'metrics': Evaluation metrics as a dictionary.
            - 'avg_inference_time': Average inference time per batch.
        loss_values (list): List of training loss values over epochs.
    """
    # Extract results
    actual = inf_result["true_values"]
    predicted = inf_result["predicted_values"]
    avg_inference_time = inf_result["avg_inference_time"]

    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

    # Compute hour-wise metrics
    metric_results = {"Hour": [], "RMSE": [], "MAE": []}
    for i in range(predicted.shape[1]):  # Assuming predicted is [samples, horizons]
        hour_actual = actual[:, i]
        hour_pred = predicted[:, i]
        # Replace this with computed metrics for each hour
        hour_metrics = {
            "RMSE": ((hour_actual - hour_pred) ** 2).mean() ** 0.5,
            "MAE": abs(hour_actual - hour_pred).mean(),
        }

        metric_results["Hour"].append(i + 1)
        metric_results["RMSE"].append(hour_metrics["RMSE"])
        metric_results["MAE"].append(hour_metrics["MAE"])

    # Plot hour-wise metrics
    plot_hour_metrics(metric_results)

    # Plot predictions vs. actuals
    plot_predictions(actual, predicted)

    # Plot training loss
    plot_training_loss(loss_values)
