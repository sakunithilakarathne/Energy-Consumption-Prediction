import wandb
import os
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from src.utils.evaluation import evaluate
from config import CHRONOS_BASELINE_MODEL, ARTIFACTS_DIR

prediction_length=24
freq='H'
target_col = "PJME_MW"

# You may need to install chronos / autogluon timeseries etc
# e.g. pip install autogluon.timeseries

def train_chronos():

    run = wandb.init(project="Time_Series_Forecasting", job_type="baseline_models", name="baseline_chronos")

    raw_artifact = run.use_artifact(
        "scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3",
        type="dataset"
    )

    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")
    
    def to_tsd(df, target_col, series_id="series_1"):
        df2 = df.copy()
        df2 = df2.reset_index().rename(columns={df.index.name: "Datetime"})
        df2["item_id"] = series_id
        df2 = df2[["item_id", "Datetime", target_col]]
        df2 = df2.rename(columns={target_col: "target"})
        return TimeSeriesDataFrame.from_data_frame(df2)

    train_ts = to_tsd(train_df, target_col)
    test_ts = to_tsd(test_df, target_col)

    # ---- Step 2: Initialize Chronos Predictor ----
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric="MASE",
        freq=freq,
        presets="bolt_base"  # Chronos-Bolt pretrained model
    )

    # ---- Step 3: Train Chronos Model ----
    predictor.fit(train_ts)

    # ---- Step 4: Forecast ----
    forecasts = predictor.predict(train_ts)
    forecast_df = forecasts.to_pandas()

    # For single series, get the first column
    y_pred = forecast_df.iloc[-len(test_df):, 0].values
    y_true = test_df[target_col].values

    # ---- Step 5: Evaluate ----
    metrics = evaluate(y_true, y_pred)
    wandb.log(metrics)

    # ---- Step 6: Plot & Log ----
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    test_df[target_col].plot(label="Actual", color="black")
    pd.Series(y_pred, index=test_df.index).plot(label="Chronos Forecast", color="tab:blue")
    plt.legend()
    plt.title("Amazon Chronos Forecast vs Actual")
    wandb.log({"chronos_forecast_plot": wandb.Image(plt)})

    # ---- Step 7: Save Model ----
    model_dir = "chronos_model"
    predictor.save(model_dir)
    artifact = wandb.Artifact("chronos_model", type="model")
    artifact.add_dir(model_dir)
    run.log_artifact(artifact)

    print("✅ Chronos training complete. Metrics:", metrics)
