import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import wandb
import numpy as np
import pandas as pd
import json
import random
import os
from src.utils.evaluation import evaluate
from config import BEST_LSTM_PARAMS

n_trials=10
lookback=168
horizon=24
target_col = "PJME_MW"

def tune_lstm():

    run = wandb.init(project="Time_Series_Forecasting", job_type="hyperparameter_tuning", name= "lstm_hp_tuning")

    raw_artifact = run.use_artifact(
        'scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3',
        type='dataset'
        )
    
    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")

    y = train_df[target_col].values
    y_test = test_df[target_col].values

    def create_sequences(series, lookback, horizon):
        X, y = [], []
        for i in range(len(series) - lookback - horizon):
            X.append(series[i:i+lookback])
            y.append(series[i+lookback:i+lookback+horizon])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(y, lookback, horizon)
    X_test, y_test_seq = create_sequences(np.concatenate([y[-lookback:], y_test]), lookback, horizon)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    param_space = {
        "lstm_units": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "epochs": [10, 20, 30]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for trial in range(n_trials):
        params = {k: random.choice(v) for k, v in param_space.items()}

        model = models.Sequential([
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(params["lstm_units"], return_sequences=False),
            layers.Dropout(params["dropout"]),
            layers.Dense(horizon)
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
        model.compile(optimizer=opt, loss='mse')

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0
        )

        y_pred = model.predict(X_test).flatten()
        y_true = y_test_seq.flatten()
        metrics = evaluate(y_true, y_pred)

        wandb.log({**params, **metrics})

        if metrics['RMSE'] < best_score:
            best_score = metrics['RMSE']
            best_params = params
            best_metrics = metrics

    # Save best parameters
    with open(BEST_LSTM_PARAMS, "w") as f:
        json.dump(best_params, f, indent=4)

    artifact = wandb.Artifact("best_lstm_params", type="parameters")
    artifact.add_file(BEST_LSTM_PARAMS)
    run.log_artifact(artifact)
    wandb.finish() 

    print("✅ Best LSTM Params:", best_params)
    print("📈 Best LSTM Metrics:", best_metrics)
    
