"""Daily BTC ETL + retraining pipeline.

fetch_data -> preprocess_features -> train_and_evaluate -> check_threshold -> reload_api
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task

API_URL = os.getenv("API_URL", "http://api:5000")
API_KEY = os.getenv("API_KEY", "")
POSTGRES_CONN_ID = "btc_postgres"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS btc_prices (
    date DATE PRIMARY KEY,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
    close DOUBLE PRECISION, volume DOUBLE PRECISION,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

default_args = {
    "owner": "faizan",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Fetch BTC-USD daily data, engineer features, retrain 7-day model, reload API",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["btc", "mlops", "etl"],
) as dag:

    @task()
    def fetch_data() -> dict:
        """Extract: pull daily BTC-USD OHLCV and store in CSV + Postgres."""
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        from src.config import RAW_DATA_PATH
        from src.etl_fetch import fetch_btc_data, save_raw

        df = fetch_btc_data()
        save_raw(df)

        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run(CREATE_TABLE_SQL)
        rows = [
            (
                str(r.Date),
                float(r.Open),
                float(r.High),
                float(r.Low),
                float(r.Close),
                float(r.Volume),
            )
            for r in df.itertuples(index=False)
        ]
        hook.insert_rows(
            table="btc_prices",
            rows=rows,
            target_fields=["date", "open", "high", "low", "close", "volume"],
            replace=True,
            replace_index="date",
            commit_every=1000,
        )
        return {
            "raw_path": str(RAW_DATA_PATH),
            "rows": len(df),
            "last_date": str(df["Date"].iloc[-1]),
        }

    @task()
    def preprocess_features(fetch_info: dict) -> dict:
        """Transform: clean data, add moving averages + volatility features."""
        import pandas as pd

        from src.config import FEATURES_PATH
        from src.features import clean_and_engineer

        raw = pd.read_csv(fetch_info["raw_path"])
        feats = clean_and_engineer(raw)
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(FEATURES_PATH, index=False)
        last = feats.iloc[-1]
        print(
            f"Rows={len(feats)} | last close={last.Close:.2f} | MA7={last.ma_7:.2f} | "
            f"MA30={last.ma_30:.2f} | vol7={last.volatility_7:.4f}"
        )
        return {"features_path": str(FEATURES_PATH), "rows": len(feats)}

    @task()
    def train_and_evaluate(feature_info: dict) -> dict:
        """Load: fit model, evaluate on last 7 days, register if it passes threshold."""
        from pathlib import Path

        from src.train_forecast import train

        metrics = train(raw_path=Path(feature_info["features_path"]))
        return {k: metrics[k] for k in ["MAE", "RMSE", "MAPE", "model_version", "passed_threshold"]}

    @task.short_circuit()
    def check_threshold(metrics: dict) -> bool:
        """Skip the API reload if the new model was not registered."""
        print(f"MAPE={metrics['MAPE']}%  passed={metrics['passed_threshold']}")
        return bool(metrics["passed_threshold"])

    @task(retries=3, retry_delay=timedelta(seconds=30))
    def reload_api(metrics: dict) -> dict:
        """Notify the Flask service to re-read the updated model weights."""
        import requests

        resp = requests.post(f"{API_URL}/reload", headers={"X-API-Key": API_KEY}, timeout=30)
        resp.raise_for_status()
        print("Reload response:", resp.json())

        forecast = requests.post(f"{API_URL}/predict", json={"forecast_days": 7}, timeout=30)
        forecast.raise_for_status()
        body = forecast.json()
        for p in body["predictions"]:
            print(f"  {p['date']}: {p['predicted_close']:.2f}")
        return {"model_version": body["model_version"], "trained_version": metrics["model_version"]}

    fetched = fetch_data()
    features = preprocess_features(fetched)
    metrics = train_and_evaluate(features)
    check_threshold(metrics) >> reload_api(metrics)
