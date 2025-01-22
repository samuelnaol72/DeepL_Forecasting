# Deep Learning Time Forecasting Library

## Overview
This library provides a framework for time series forecasting using a variety of deep learning models, including LSTMs, GRUs, CNN-LSTMs, and the N-BEATS model. It includes modules for data preprocessing, model training, evaluation, and visualization, enabling a seamless workflow for time series forecasting tasks.

---

## Features

### Models Supported
- **LSTM**: Long Short-Term Memory Networks for sequential data.
- **GRU**: Gated Recurrent Units for efficient sequence modeling.
- **CNN-LSTM**: Combination of Convolutional Neural Networks and LSTMs.
- **LSTM41**: A specialized LSTM variant.
- **LSTMRESET**: An LSTM with hidden state reset capabilities.
- **N-BEATS**: Neural Basis Expansion Analysis for Interpretable Time Series.

### Key Functionalities
- **Data Preprocessing**:
  - Handle outliers.
  - Create lagged features and future targets.
  - Split datasets into training and validation sets.
- **Model Training**:
  - Flexible configuration via YAML files.
  - Learning rate scheduler (e.g., ReduceLROnPlateau).
  - Logs training and validation metrics.
- **Evaluation**:
  - Comprehensive metrics, including RMSE, MAE, Precision, and Recall.
  - Time performance analysis.
- **Visualization**:
  - Plot training loss trends.
  - Compare predictions vs. actuals.
  - Visualize evaluation metrics.

---

## Project Structure

```
Library/
├── configs/          # Configuration files (YAML)
├── data_provider/    # Data preprocessing and loading
├── docs/             # Documentation and presentations
├── example/          # Example scripts and datasets
│   ├── dataset/      # Sample datasets
│   ├── results/      # Output directory for results
├── models/           # Implementation of supported models
├── run/              # Training and testing scripts
├── utils/            # Utility functions (logging, saving, visualization)
└── requirements.txt  # Python dependencies
```

---

## Setup

### Prerequisites
- Python 3.8 or later
- Conda or virtual environment manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/samuelnaol72/DeepL_Forecasting.git
   cd DeepL_Forecasting
   ```
2. Create a virtual environment:
   ```bash
   conda create --name forecasting_env python=3.8
   conda activate forecasting_env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Configuration
Modify the configuration file `configs/config_lstm.yaml` to adjust the settings for preprocessing, model training, and evaluation.

### Training
Run the training script to train a selected model:
```bash
python example/lstm_test.py
```

### Testing
Evaluate the trained model using the test script:
```bash
python example/lstm_test.py
```

---

## Key Files

### `configs/config_lstm.yaml`
A YAML file that defines hyperparameters for data preprocessing, model training, and evaluation.

### `models/`
Contains implementations of supported models (LSTM, GRU, CNN-LSTM, N-BEATS, etc.).

### `run/train.py`
Script to train models with specified configurations.

### `run/test.py`
Script to evaluate trained models and compute metrics.

---

## Example Output
- **Training Loss Trend**: Visualizes loss reduction over epochs.
- **Evaluation Metrics**: RMSE, MAE, Precision, Recall.
- **Prediction vs. Actuals**: Plots showing how well the model predicts future values.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any feature additions or bug fixes.
---

## Contact
For any inquiries, reach out to [samuelnaol72@gmail.com](mailto:samuelnaol72@gmail.com).

