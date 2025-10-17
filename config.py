import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATASET = os.path.join(RAW_DATA_DIR, "PJME_hourly.csv")
PROCESSED_DATASET = os.path.join(PROCESSED_DATA_DIR, "processed_PJME.csv")

TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

TRAIN_PARQUET = os.path.join(ARTIFACTS_DIR, "train.parquet")
TEST_PARQUET = os.path.join(ARTIFACTS_DIR, "test.parquet")

PROPHET_BASELINE_MODEL = os.path.join(ARTIFACTS_DIR,"baseline_prophet_model.pkl")
LSTM_BASELINE_MODEL = os.path.join(ARTIFACTS_DIR,"baseline_lstm_model.h5")
CHRONOS_BASELINE_MODEL = os.path.join(ARTIFACTS_DIR,"chronos_predictor.pkl")
NBEATS_BASELINE_MODEL = os.path.join(ARTIFACTS_DIR, "nbeats_model.pth")

BEST_PROPHET_PARAMS = os.path.join(ARTIFACTS_DIR, "best_prophet_params.json")
BEST_LSTM_PARAMS = os.path.join(ARTIFACTS_DIR, "best_lstm_params.json")
