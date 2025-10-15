from prophet import Prophet
import itertools
import json
import wandb
import numpy as np
import pandas as pd
from src.utils.evaluation import evaluate
from config import BEST_PROPHET_PARAMS


target_col = "PJME_MW"
horizon = 'H'

def tune_prophet():

    run = wandb.init(project="Time_Series_Forecasting", job_type="hyperparameter_tuning", name= "prophet_hp_tuning")

    raw_artifact = run.use_artifact(
        'scsthilakarathne-nibm/Time_Series_Forecasting/PJME_Datasets:v3',
        type='dataset'
        )
    
    artifact_dir = raw_artifact.download()
    train_df = pd.read_parquet(f"{artifact_dir}/train.parquet")
    test_df = pd.read_parquet(f"{artifact_dir}/test.parquet")
  
    df_train = train_df.reset_index().rename(columns={'Datetime': 'ds', target_col: 'y'})
    df_test = test_df.reset_index().rename(columns={'Datetime': 'ds', target_col: 'y'})

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    best_score = np.inf
    best_params = None
    best_metrics = None

    for params in all_params:
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(df_train)
        future = model.make_future_dataframe(periods=len(df_test), freq=horizon)
        forecast = model.predict(future)
        pred = forecast.set_index('ds').loc[df_test['ds'], 'yhat']

        y_true = df_test['y'].values
        y_pred = pred.values
        metrics = evaluate(y_true, y_pred)

        wandb.log({**params, **metrics})

        if metrics['RMSE'] < best_score:
            best_score = metrics['RMSE']
            best_params = params
            best_metrics = metrics

    # Save best parameters
    with open(BEST_PROPHET_PARAMS, "w") as f:
        json.dump(best_params, f, indent=4)

    artifact = wandb.Artifact("best_prophet_params", type="parameters")
    artifact.add_file(BEST_PROPHET_PARAMS)
    run.log_artifact(artifact)
    wandb.finish()

    print("✅ Best Prophet Params:", best_params)
    print("📈 Best Prophet Metrics:", best_metrics)
    
    
