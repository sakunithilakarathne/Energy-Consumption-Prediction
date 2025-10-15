import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import joblib
import os
import wandb
import numpy as np
import pandas as pd
from src.utils.evaluation import evaluate
from config import ARTIFACTS_DIR, LSTM_BASELINE_MODEL


target_col = "PJME_MW"
lookback=168
horizon=24
epochs=10
batch_size=64


def train_lstm():

    run = wandb.init(project="Time_Series_Forecasting", job_type="baseline_models", name="baseline_lstm")

    raw_artifact = run.use_artifact(
        "scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3",
        type="dataset"
    )

    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")

    # Convert target to numpy arrays
    y = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Create sequences
    def create_sequences(series, lookback, horizon):
        X, y = [], []
        for i in range(len(series) - lookback - horizon):
            X.append(series[i:i+lookback])
            y.append(series[i+lookback:i+lookback+horizon])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(y, lookback, horizon)
    X_test, y_test_seq = create_sequences(np.concatenate([y[-lookback:], y_test]), lookback, horizon)

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Build model
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Predict
    y_pred = model.predict(X_test).flatten()
    y_true = y_test_seq.flatten()
    
    # Evaluate
    metrics = evaluate(y_true, y_pred)
    run.log(metrics)
    
    # Log training history
    run.log({
        "lstm_train_loss": history.history['loss'][-1],
        "lstm_val_loss": history.history['val_loss'][-1]
    })
    

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save(LSTM_BASELINE_MODEL)
    
    artifact = wandb.Artifact("lstm_model", type="model", description="Baseline LSTM Model")
    artifact.add_file(LSTM_BASELINE_MODEL)
    run.log_artifact(artifact)
    
    print("✅ LSTM training complete. Metrics:", metrics)
    
    wandb.finish()
