"""Central paths & settings (overridable through environment variables)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

RAW_DATA_PATH = DATA_DIR / "raw" / "btc_raw.csv"
FEATURES_PATH = DATA_DIR / "processed" / "btc_features.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TICKER = os.getenv("TICKER", "BTC-USD")
START_DATE = os.getenv("START_DATE", "2018-01-01")
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))

# Model is registered only if MAPE on the 7-day test window is below this
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "10.0"))
