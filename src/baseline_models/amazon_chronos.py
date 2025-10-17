import wandb
import os
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from src.utils.evaluation import evaluate
from config import CHRONOS_BASELINE_MODEL, ARTIFACTS_DIR
import matplotlib.pyplot as plt

prediction_length=24
freq='H'
target_col = "PJME_MW"

# You may need to install chronos / autogluon timeseries etc
# e.g. pip install autogluon.timeseries

def train_chronos():

    run = wandb.init(project="Time_Series_Forecasting",
                     job_type="baseline_models",
                     name="baseline_chronos")

    # ---- Step 1: Load data from W&B artifact ----
    raw_artifact = run.use_artifact(
        "scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3",
        type="dataset"
    )
    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")

    # ---- Step 2: Convert to TimeSeriesDataFrame ----
    def to_tsd(df, target_col, series_id="series_1"):
        df2 = df.copy()
        # Rename datetime column correctly
        if "Datetime" in df2.columns:
            df2 = df2.rename(columns={"Datetime": "timestamp"})
        elif "datetime" in df2.columns:
            df2 = df2.rename(columns={"datetime": "timestamp"})
        else:
            raise ValueError("No Datetime column found.")
        df2["item_id"] = series_id
        df2 = df2[["item_id", "timestamp", target_col]].rename(columns={target_col: "target"})
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        return TimeSeriesDataFrame.from_data_frame(df2)

    train_ts = to_tsd(train_df, target_col)
    test_ts = to_tsd(test_df, target_col)

    # ---- Step 3: Initialize Predictor ----
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric="MASE"
    )

    # ---- Step 4: Train Chronos model ----
    predictor.fit(train_ts, presets="medium_quality", time_limit=600)  # <= 10 minutes

    # ---- Step 5: Forecast ----
    forecasts = predictor.predict(test_ts)
    forecast_df = forecasts.to_pandas()

    # For single series, get values
    y_pred = forecast_df.iloc[:, 0].values
    y_true = test_df[target_col].values

    # ---- Step 6: Evaluate ----
    metrics = evaluate(y_true, y_pred)
    wandb.log(metrics)

    # ---- Step 7: Plot & Log ----
    plt.figure(figsize=(10, 4))
    test_df[target_col].plot(label="Actual", color="black")
    pd.Series(y_pred, index=test_df.index).plot(label="Chronos Forecast", color="tab:blue")
    plt.legend()
    plt.title("Amazon Chronos Forecast vs Actual")
    wandb.log({"chronos_forecast_plot": wandb.Image(plt)})

    # ---- Step 8: Save Model ----
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    predictor.save(CHRONOS_BASELINE_MODEL)
    artifact = wandb.Artifact("chronos_model", type="model", description="Baseline chronos model")
    artifact.add_dir(CHRONOS_BASELINE_MODEL)
    run.log_artifact(artifact)

    print("✅ Chronos training complete. Metrics:", metrics)