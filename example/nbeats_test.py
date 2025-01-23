import yaml
import pandas as pd
from data_provider.data_factory import create_loaders
from run.train import train_model
from run.test import test_model
from utils.viz import visualize_results
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def load_config(config_path="configs/config_nbeats.yaml"):
    """Loads the YAML configuration file for the NBeats model."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Load the dataset using the path from the config
    dataset_path = config["dataset"]["path"]
    df = pd.read_parquet(dataset_path)
    print("Loaded dataset:", df.head())

    # Create the train and validation data loaders
    train_loader, val_loader = create_loaders(df, config)

    # Train the NBeats model
    print("Training the NBeats model...")
    trained_model, loss_trend = train_model(config, train_loader, val_loader)

    # Test the NBeats model
    print("Testing the NBeats model...")
    inf_result = test_model(config, trained_model, val_loader)
    print("Inference Results:", inf_result["metrics"])
    """
    test_model:
    # return {
            "true_values": y_true,
            "predicted_values": y_pred,
            "metrics": combined_metrics,
            "avg_inference_time": avg_inference_time,
        }
    """

    # Visualization
    visualize_results(inf_result, loss_trend)
