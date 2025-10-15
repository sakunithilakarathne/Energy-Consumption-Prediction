import pandas as pd
import wandb
from config import RAW_DATASET
from src.preprocessing.preprocessing import preprocess_time_series

def main():

    dataset = pd.read_csv(RAW_DATASET)
    preprocess_time_series(
        df=dataset, 
        datetime_col="Datetime",
        target_col="PJME_MW",
        freq="H",
        forecast_horizon=24,
        test_size=24*30,
        clip_outliers=True,
        scale_features=True,
        project_name= "Time_Series_Forecasting",
        upload_to_wandb=True,
        verbose=True
        )





if __name__ == "__main__":
    main()