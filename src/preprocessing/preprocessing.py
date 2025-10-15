import pandas as pd
import numpy as np
import pickle
import wandb
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from config import ARTIFACTS_DIR, TRAIN_PATH, TEST_PATH, SCALER_PATH, PROCESSED_DATASET

def preprocess_time_series(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    freq: str = 'H',
    forecast_horizon: int = 24,
    test_size: int = 24 * 30,  # last 30 days
    clip_outliers: bool = True,
    scale_features: bool = True,
    project_name: str = "Time_Series_Forecasting",
    upload_to_wandb: bool = True,
    verbose: bool = True
):


    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Step 1: Set datetime index ---
    df = df.copy()

    if verbose: print("🔹 Identifying duplicates...")
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # --- Handle duplicate timestamps before setting index ---
    dup_count = df.duplicated(subset=[datetime_col]).sum()
    if dup_count > 0:
        if verbose:
            print(f"Found {dup_count} duplicate timestamps. Aggregating by mean...")
        df = df.groupby(datetime_col, as_index=False).mean(numeric_only=True)

    if verbose: print("🔹 Setting datetime index...")
    df = df.set_index(datetime_col).sort_index()

    # Reindex to uniform frequency safely
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq(freq)

    # --- Step 2: Handle missing values ---
    if verbose: print("🔹 Handling missing values...")
    df[target_col] = df[target_col].interpolate(method='time')
    df[target_col] = df[target_col].fillna(method='bfill').fillna(method='ffill')

    # --- Step 3: Clip outliers ---
    if clip_outliers:
        if verbose: print("🔹 Clipping extreme outliers...")
        z = (df[target_col] - df[target_col].mean()) / df[target_col].std()
        df[target_col] = df[target_col].where(z.abs() < 6, np.nan)
        df[target_col] = df[target_col].interpolate(method='time')

    # --- Step 4: Create time-based features ---
    if verbose: print("🔹 Creating time-based features...")
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['dayofyear'] = df.index.dayofyear

    # --- Step 5: Lag features ---
    if verbose: print("🔹 Creating lag features...")
    lags = [1, 24, 168] if freq == 'H' else [1, 7, 30]
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # --- Step 6: Rolling statistics ---
    if verbose: print("🔹 Creating rolling statistics...")
    df['roll_mean_24'] = df[target_col].shift(1).rolling(window=24).mean()
    df['roll_std_24'] = df[target_col].shift(1).rolling(window=24).std()
    df['roll_mean_168'] = df[target_col].shift(1).rolling(window=168).mean()

    df = df.dropna()

    # Savve processed dataset
    df.to_csv(PROCESSED_DATASET)

    # --- Step 7: Train/test split ---
    if verbose: print("🔹 Splitting train/test...")
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    # --- Step 8: Scale features ---
    scaler = None
    if scale_features:
        if verbose: print("🔹 Scaling features...")
        scaler = StandardScaler()
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        train.loc[:, numeric_cols] = scaler.fit_transform(train[numeric_cols])
        test.loc[:, numeric_cols] = scaler.transform(test[numeric_cols])

    # --- Step 9: Save locally ---
    if verbose: print(f"💾 Saving artifacts locally to: {ARTIFACTS_DIR}")


    train.to_csv(TRAIN_PATH)
    test.to_csv(TEST_PATH)
    if scaler:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    # --- Step 10: Upload to Weights & Biases ---
    if upload_to_wandb:
        if verbose: print("🚀 Uploading to Weights & Biases...")

        # Initialize W&B run
        wandb.init(project=project_name, job_type="data_preprocessing", name= "data_preprocessing")

        artifact = wandb.Artifact(
            name=f"PJME_Datasets",
            type="dataset",
            description="Preprocessed time series data with train/test splits and scaler",
            metadata={
                "freq": freq,
                "forecast_horizon": forecast_horizon,
                "test_size": test_size,
                "clip_outliers": clip_outliers,
                "scale_features": scale_features,
            },
        )

        # Add files
        artifact.add_file(TRAIN_PATH)
        artifact.add_file(TEST_PATH)
        if os.path.exists(SCALER_PATH):
            artifact.add_file(SCALER_PATH)

        wandb.log_artifact(artifact)
        wandb.finish()

        if verbose: print("✅ W&B artifact upload complete.")

    if verbose:
        print("✅ Preprocessing complete.")
        print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    return train, test, scaler
