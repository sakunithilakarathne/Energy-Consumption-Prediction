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
    
    def to_tsd(df, target_col, id_col="series1"):
        df2 = df.reset_index().rename(columns={df.index.name: "timestamp", target_col: "target"})
        df2["item_id"] = id_col
        return TimeSeriesDataFrame.from_data_frame(df2, id_col="item_id", timestamp_col="timestamp", target_col="target")

    train_ts = to_tsd(train_df, target_col)
    test_ts = to_tsd(test_df, target_col)

    # Initialize predictor; use preset “bolt_base” (Chronos-Bolt) or other models
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric="MASE",       # or "RMSE", "MAPE" depending on support
        presets="bolt_base"       # using Chronos-Bolt by default
    )

    predictor = predictor.fit(train_ts)

    # Get forecasts
    forecasts = predictor.predict(train_ts)  # this returns a TimeSeriesDataFrame

    # Extract the forecast for the test period
    # Convert to pandas series aligned to test_df timestamps
    fut = forecasts.drop_columns(["item_id"]).to_pandas()  # shape: (timestamps x items)
    # Assume single series
    pred_series = fut.iloc[-len(test_df):, 0]
    y_true = test_df[target_col].values
    y_pred = pred_series.values

    metrics = evaluate(y_true, y_pred)
    run.log(metrics)

    # Optionally log forecast plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    test_df[target_col].plot(label="true")
    pd.Series(y_pred, index=test_df.index).plot(label="chronos_pred")
    plt.legend()
    plt.title("Chronos Forecast vs Actual")
    run.log({"chronos_forecast_plot": wandb.Image(plt)})

    # Save predictor/trained model
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    predictor.save(CHRONOS_BASELINE_MODEL)
    artifact = wandb.Artifact("chronos_model", type="model", description="Baseline Chronos Model")
    artifact.add_file(CHRONOS_BASELINE_MODEL)
    run.log_artifact(artifact)

    print("✅ Chronos training complete. Metrics:", metrics)
