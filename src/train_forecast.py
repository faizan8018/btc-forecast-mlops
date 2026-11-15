"""Train the 7-day BTC close-price forecaster and save model artifacts.

Model : StandardScaler + Ridge regression on lag / moving-average /
        volatility features, forecasting 7 days recursively.
Eval  : last 7 days are held out; the model trained on earlier data
        forecasts them recursively -> MAE, RMSE, MAPE.
Usage :  python -m src.train_forecast
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FORECAST_DAYS, MAPE_THRESHOLD, METRICS_PATH, MODEL_PATH, RAW_DATA_PATH
from src.features import FEATURE_COLS, make_supervised, recursive_forecast, regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_forecast")


def save_artifact(artifact: dict, path) -> None:
    """Save the model with Python's standard C pickler, atomically.

    Airflow imports `dill`, which patches the pure-Python pickler that
    joblib.dump uses, so files saved that way need `dill` to load. The C
    pickler is not affected, so the model loads anywhere scikit-learn exists.
    Writing to a temp file + os.replace means the API never reads a half-written file.
    """
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def new_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def evaluate(close: pd.Series, horizon: int = FORECAST_DAYS) -> dict:
    """Hold out the last `horizon` days and forecast them recursively."""
    train_close, test_close = close.iloc[:-horizon], close.iloc[-horizon:]
    X, y = make_supervised(train_close)
    model = new_model().fit(X, y)
    preds = recursive_forecast(model, train_close.values, horizon)
    metrics = regression_metrics(test_close.values, preds)
    metrics["test_window"] = [str(d.date()) for d in test_close.index]
    metrics["actual"] = [round(float(v), 2) for v in test_close.values]
    metrics["predicted"] = [round(v, 2) for v in preds]
    return metrics


def train(raw_path=RAW_DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH) -> dict:
    df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date")
    close = df.set_index("Date")["Close"].astype(float)
    log.info(
        "Loaded %d daily closes (%s to %s)",
        len(close),
        close.index[0].date(),
        close.index[-1].date(),
    )

    metrics = evaluate(close)
    passed = metrics["MAPE"] < MAPE_THRESHOLD
    log.info(
        "Validation on last %d days: MAE=%.2f RMSE=%.2f MAPE=%.3f%% -> %s",
        FORECAST_DAYS,
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        "PASSED" if passed else "FAILED",
    )

    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    metrics.update(
        {
            "model_version": version,
            "model_type": "Ridge (lag + MA + volatility features)",
            "mape_threshold": MAPE_THRESHOLD,
            "passed_threshold": passed,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(close)),
        }
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if passed:
        X, y = make_supervised(close)  # refit on ALL data for production
        model = new_model().fit(X, y)
        artifact = {
            "model": model,
            "version": version,
            "feature_cols": FEATURE_COLS,
            "last_date": str(close.index[-1].date()),
            "history": close.values[-60:].tolist(),
        }
        save_artifact(artifact, model_path)
        log.info("Model %s registered -> %s", version, model_path)
    else:
        log.warning("Model NOT registered (MAPE above threshold); keeping previous model")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved -> %s", metrics_path)
    return metrics


if __name__ == "__main__":
    train()
