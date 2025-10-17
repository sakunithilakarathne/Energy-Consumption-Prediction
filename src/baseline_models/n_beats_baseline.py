import wandb
import os
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae as darts_mae, rmse as darts_rmse, mape as darts_mape
from darts.metrics import mape as darts_mape_fn
from src.utils.evaluation import evaluate
from config import NBEATS_BASELINE_MODEL, ARTIFACTS_DIR

input_chunk_length=168
output_chunk_length=24
n_epochs=5
target_col = "PJME_MW"


def train_nbeats():

    run = wandb.init(project="Time_Series_Forecasting", job_type="baseline_models", name="baseline_nbeats")

    raw_artifact = run.use_artifact(
        "scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3",
        type="dataset"
    )
    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")

    # Convert to Darts TimeSeries
    ts_full = TimeSeries.from_series(train_df[target_col])
    ts_test = TimeSeries.from_series(test_df[target_col])

    # Split train / test (we assume train_df + test_df contiguous)
    ts_train = ts_full  # if train_df covers full history minus test
    # Forecast horizon = len(test_df)
    horizon = len(test_df)

    # Initialize model
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        batch_size = 32,
        random_state=42
    )

    # Train
    model.fit(ts_train)

    # Predict
    pred_ts = model.predict(horizon)

    # Evaluate: align with test
    # Convert to pandas
    y_true = ts_test.values().ravel()
    y_pred = pred_ts.values().ravel()

    # Using your evaluate()
    metrics = evaluate(y_true, y_pred)
    run.log(metrics)

    # Also log Darts metrics for sanity
    darts_metrics = {
        "darts_MAE": darts_mae(ts_test, pred_ts),
        "darts_RMSE": darts_rmse(ts_test, pred_ts),
        "darts_MAPE": darts_mape(ts_test, pred_ts),
    }
    run.log(darts_metrics)

    # Plot forecast vs actual
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    test_df[target_col].plot(label="actual")
    pd.Series(y_pred, index=test_df.index).plot(label="nbeats_pred")
    plt.legend()
    plt.title("N-BEATS Forecast")
    run.log({"nbeats_forecast_plot": wandb.Image(plt)})

    # Save model
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save(NBEATS_BASELINE_MODEL)
    artifact = wandb.Artifact("nbeats_model", type="model", description="Baseline N-BEATS Model")
    artifact.add_file(NBEATS_BASELINE_MODEL)
    run.log_artifact(artifact)

    print("✅ N-BEATS training complete. Metrics:", metrics)
