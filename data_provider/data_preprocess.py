import pandas as pd
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# Timing decorator
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f"Execution time for {func.__name__}: {end_time - start_time:.2f}s"
        )
        return result

    return wrapper


# Logging decorator (renamed to avoid conflict)
def log_decorator(func):
    def preprocessing_func(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args[1:]}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            logging.info(
                f"{func.__name__} returned: {[res.shape if hasattr(res, 'shape') else len(res) for res in result]}"
            )
        else:
            logging.info(f"{func.__name__} returned: {result}")
        return result

    return preprocessing_func


def preprocessing_g1(df, num_lags, lookhour, id):
    """
    Preprocesses the data for a single ID by handling outliers, adding time-based features,
    and creating lagged (past) and future target features.

    Parameters:
        df (pd.DataFrame): Input dataframe containing raw data.
        num_lags (int): Number of lagged features to create (e.g., previous hours' data).
        lookhour (int): Number of future hours to predict (future target features).
        id (int): Unique identifier for the dataset (e.g., station ID or device ID).

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - Y_train (pd.DataFrame): Training targets.
            - X_val (pd.DataFrame): Validation features.
            - Y_val (pd.DataFrame): Validation targets.
    """

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter data by id
    df = df[df["id"] == id]

    # Outlier removal
    columns_to_clean = ["atm_temp", "humi", "pres", "obj_temp"]
    thresholds = {
        "obj_temp": (-20, 45),
        "atm_temp": (-20, 45),
        "humi": (0, 100),
        "pres": (900, 1100),
    }
    for column in columns_to_clean:
        lower, upper = thresholds[column]
        df[column] = df[column].where(
            (df[column] >= lower) & (df[column] <= upper), np.nan
        )

    # Add time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour

    # Sampling data hourly
    result_df = df.groupby(["year", "month", "day", "hour"]).first().reset_index()
    start_date = result_df["date"].min()
    end_date = result_df["date"].max()
    full_range = pd.date_range(start=start_date, end=end_date, freq="h")
    full_range_df = pd.DataFrame(
        {
            "year": full_range.year,
            "month": full_range.month,
            "day": full_range.day,
            "hour": full_range.hour,
        }
    )

    # Merge full range with sampled data
    merged_df = pd.merge(
        full_range_df, result_df, on=["year", "month", "day", "hour"], how="left"
    )
    merged_df["serialnum"] = id

    # Create lagged and future features
    df_X_cols = ["year", "month", "day", "hour", "serialnum"] + columns_to_clean
    df_Y_cols = []

    # Lagged features
    for i in range(1, 1 + num_lags):
        merged_df[f"obj_before({i})"] = merged_df["obj_temp"].shift(i)
        df_X_cols.append(f"obj_before({i})")

    # Future features (targets)
    for i in range(1, 1 + lookhour):
        merged_df[f"obj_after({i})"] = merged_df["obj_temp"].shift(-i)
        df_Y_cols.append(f"obj_after({i})")

    # Remove rows with missing values
    merged_df.dropna(inplace=True)

    # Split data into training and validation sets
    split_index = int(len(merged_df) * 0.8)
    df_train = merged_df.iloc[:split_index]
    df_test = merged_df.iloc[split_index:]

    X_train = df_train[df_X_cols]
    Y_train = df_train[df_Y_cols]
    X_val = df_test[df_X_cols]
    Y_val = df_test[df_Y_cols]

    return X_train, Y_train, X_val, Y_val


@timing
@log_decorator
def preprocessing(df, num_lags, lookhour, ids):
    """
    Preprocesses the data for multiple IDs by applying the preprocessing_g1 function
    to each ID and combining the results.

    Parameters:
        df (pd.DataFrame): Input dataframe containing raw data.
        num_lags (int): Number of lagged features to create (e.g., previous hours' data).
        lookhour (int): Number of future hours to predict (future target features).
        ids (list): List of unique identifiers (e.g., station IDs or device IDs).

    Returns:
        tuple: A tuple containing:
            - X_train_combined (pd.DataFrame): Combined training features for all IDs.
            - Y_train_combined (pd.DataFrame): Combined training targets for all IDs.
            - X_val_combined (pd.DataFrame): Combined validation features for all IDs.
            - Y_val_combined (pd.DataFrame): Combined validation targets for all IDs.
    """
    X_train_list = []
    Y_train_list = []
    X_val_list = []
    Y_val_list = []

    for id in ids:
        # Process each ID using preprocessing_g1
        X_train, Y_train, X_val, Y_val = preprocessing_g1(df, num_lags, lookhour, id)

        # Append results to corresponding lists
        X_train_list.append(X_train)
        Y_train_list.append(Y_train)
        X_val_list.append(X_val)
        Y_val_list.append(Y_val)

    # Concatenate all dataframes
    X_train_combined = pd.concat(X_train_list, axis=0).reset_index(drop=True)
    Y_train_combined = pd.concat(Y_train_list, axis=0).reset_index(drop=True)
    X_val_combined = pd.concat(X_val_list, axis=0).reset_index(drop=True)
    Y_val_combined = pd.concat(Y_val_list, axis=0).reset_index(drop=True)

    return X_train_combined, Y_train_combined, X_val_combined, Y_val_combined
