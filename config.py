import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed")
RAW_DATASET = os.path.join(RAW_DATA_DIR, "PJME_hourly.csv")

TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")