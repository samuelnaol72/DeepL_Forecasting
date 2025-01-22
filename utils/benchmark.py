import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def benchmark(y_test, y_pred):
    """
    Computes benchmark metrics for regression models, including RMSE, MAE,
    standard deviation of errors, and the 95th quantile of absolute errors.

    Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    std = (y_test - y_pred).std()
    quantile_95 = np.quantile((y_test - y_pred), 0.95)
    result = pd.DataFrame(
        [{"RMSE": rmse, "MAE": mae, "STD": std, "95% Quantile": quantile_95}]
    )
    return result


def metric_with_times(y_test, y_pred, train_time, inf_time):
    """
    Computes evaluation metrics for regression, including precision, recall,
    temperature trends, and inference times.

    Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.
        train_time (float): Training time in seconds.
        inf_time (float): Inference time in seconds.

    Returns:
        pd.DataFrame: A DataFrame containing computed metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    std = (y_test - y_pred).std()
    quantile_95 = np.quantile(abs(y_test - y_pred), 0.95)

    # Calculate freezing temperature prediction metrics
    condition_true_freezing = y_test < 0  # True freezing temperatures
    condition_pred_freezing = y_pred < 0  # Predicted freezing temperatures

    # Correctly predicted freezing temperatures
    correct_freezing = np.sum(
        np.logical_and(condition_true_freezing, condition_pred_freezing)
    )

    # Predicted freezing incorrectly when not freezing
    false_positive_freezing = np.sum(
        np.logical_and(y_test >= 0, condition_pred_freezing)
    )

    # Missed freezing predictions
    false_negative_freezing = np.sum(
        np.logical_and(condition_true_freezing, y_pred >= 0)
    )

    # Precision and recall for freezing temperature predictions
    precision = correct_freezing / (correct_freezing + false_positive_freezing)
    recall = correct_freezing / (correct_freezing + false_negative_freezing)

    # Calculate temperature change accuracy (rising/falling prediction)
    temp_change = y_test[1:] - y_test[:-1]
    temp_change_pred = y_pred[1:] - y_pred[:-1]
    correct_trend_predictions = sum(
        (np.multiply(temp_change, temp_change_pred) > -0.00001)
    )
    temp_trend_accuracy = correct_trend_predictions / len(temp_change) * 100

    # Compile results into a DataFrame
    result = pd.DataFrame(
        [
            {
                "RMSE": rmse,
                "MAE": mae,
                "STD": std,
                "95% Quantile": quantile_95,
                "Freezing Precision": precision,
                "Freezing Recall": recall,
                "Temperature Trend Accuracy": temp_trend_accuracy,
                "Training Time (s)": train_time,
                "Inference Time (s)": inf_time,
            }
        ]
    )
    return result


def classification_eval(y_test, y_pred, train_time, inf_time):
    """
    Computes evaluation metrics for binary classification tasks, such as freezing prediction.

    Parameters:
        y_test (array-like): True target labels (0 for non-freezing, 1 for freezing).
        y_pred (array-like): Predicted target labels (0 for non-freezing, 1 for freezing).
        train_time (float): Training time in seconds.
        inf_time (float): Inference time in seconds.

    Returns:
        pd.DataFrame: A DataFrame containing precision and recall metrics.
    """
    # Correctly predicted freezing temperatures
    true_positives = np.sum((y_pred == 1) & (y_test == 1))

    # Missed freezing predictions (false negatives)
    false_negatives = np.sum((y_test == 1) & (y_pred == 0))

    # Predicted freezing incorrectly (false positives)
    false_positives = np.sum((y_test == 0) & (y_pred == 1))

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Compile results into a DataFrame
    result = pd.DataFrame(
        [
            {
                "Freezing Precision": precision,
                "Freezing Recall": recall,
                "Training Time (s)": train_time,
                "Inference Time (s)": inf_time,
            }
        ]
    )

    return result
