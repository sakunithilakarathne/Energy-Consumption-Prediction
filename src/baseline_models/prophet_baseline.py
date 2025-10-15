from prophet import Prophet
import joblib
import os
import wandb
import pandas as pd

from src.utils.evaluation import evaluate
from config import PROPHET_BASELINE_MODEL, ARTIFACTS_DIR

target_col = "PJME_MW"
horizon = 'H'


def train_prophet():

    run = wandb.init(project="Time_Series_Forecasting", job_type="baseline_models", name= "prophet_baseline")

    raw_artifact = run.use_artifact(
        'scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3',
        type='dataset'
        )
    
    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")

    # Prepare data
    df_train = train_df.reset_index().rename(columns={'Datetime': 'ds', target_col: 'y'})
    df_test = test_df.reset_index().rename(columns={'Datetime': 'ds', target_col: 'y'})
    
    # Initialize model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Fit
    model.fit(df_train)
    
    # Forecast
    future = model.make_future_dataframe(periods=len(df_test), freq=horizon)
    forecast = model.predict(future)
    
    # Extract only test range predictions
    pred = forecast.set_index('ds').loc[df_test['ds'], 'yhat']
    y_true = df_test['y'].values
    y_pred = pred.values
    
    # Evaluate
    metrics = evaluate(y_true, y_pred)
    run.log(metrics)
    
    # Plot sample forecast
    fig = model.plot(forecast)
    run.log({"prophet_forecast_plot": wandb.Image(fig)})
    
    # Save model
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, PROPHET_BASELINE_MODEL)
    
    model_artifact = wandb.Artifact("prophet_model", type="model", description= "Baseline Prophet Model")
    model_artifact.add_file(PROPHET_BASELINE_MODEL)
    run.log_artifact(model_artifact)
    
    print("✅ Prophet training complete. Metrics:", metrics)
    
    wandb.finish()
