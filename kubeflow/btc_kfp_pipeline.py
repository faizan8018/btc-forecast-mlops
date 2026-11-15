"""Kubeflow Pipeline (KFP v2) for BTC 7-day forecasting.

Three discrete *container components*, all running the project Docker image
built by GitHub Actions (it already contains src/ and every ML dependency):

    ingest_data  ->  generate_features  ->  train_model

Compile:  python kubeflow/btc_kfp_pipeline.py   ->  kubeflow/btc_kfp_pipeline.yaml
"""

from pathlib import Path

from kfp import compiler, dsl

IMAGE = "faizandocker387/btc-forecast-api:latest"

INGEST_SCRIPT = """
import sys, os
from src.etl_fetch import fetch_btc_data
out = sys.argv[1]
os.makedirs(os.path.dirname(out), exist_ok=True)
df = fetch_btc_data()
df.to_csv(out, index=False)
print(f"Ingested {len(df)} rows -> {out}")
"""

FEATURES_SCRIPT = """
import sys, os
import pandas as pd
from src.features import clean_and_engineer
raw_path, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
feats = clean_and_engineer(pd.read_csv(raw_path))
feats.to_csv(out, index=False)
print(f"Features: {list(feats.columns)} rows={len(feats)}")
"""

TRAIN_SCRIPT = """
import sys, os
from pathlib import Path
os.environ["MAPE_THRESHOLD"] = sys.argv[4]
from src.train_forecast import train
m = train(raw_path=Path(sys.argv[1]), model_path=Path(sys.argv[2]), metrics_path=Path(sys.argv[3]))
print(f"MAE={m['MAE']} RMSE={m['RMSE']} MAPE={m['MAPE']}% passed={m['passed_threshold']}")
"""


@dsl.container_component
def ingest_data(raw_data: dsl.Output[dsl.Dataset]):
    """Pull daily BTC-USD OHLCV (yfinance, Binance fallback)."""
    return dsl.ContainerSpec(
        image=IMAGE, command=["python", "-c", INGEST_SCRIPT], args=[raw_data.path]
    )


@dsl.container_component
def generate_features(raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]):
    """Clean data + 7/30-day moving averages, volatility, returns."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-c", FEATURES_SCRIPT],
        args=[raw_data.path, features.path],
    )


@dsl.container_component
def train_model(
    features: dsl.Input[dsl.Dataset],
    mape_threshold: float,
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Artifact],
):
    """Fit Ridge forecaster, evaluate on last 7 days, save model + metrics artifacts."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-c", TRAIN_SCRIPT],
        args=[features.path, model.path, metrics.path, mape_threshold],
    )


@dsl.pipeline(
    name="btc-7day-forecast-pipeline",
    description="Ingest BTC-USD data, engineer features and train a 7-day forecasting model",
)
def btc_forecast_pipeline(mape_threshold: float = 10.0):
    ingest = ingest_data()
    ingest.set_caching_options(False)  # always fetch the latest prices

    feats = generate_features(raw_data=ingest.outputs["raw_data"])

    trainer = train_model(features=feats.outputs["features"], mape_threshold=mape_threshold)
    for step in (ingest, feats, trainer):
        step.set_cpu_limit("500m").set_memory_limit("512M")


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".yaml")
    compiler.Compiler().compile(pipeline_func=btc_forecast_pipeline, package_path=str(out))
    print(f"Compiled pipeline -> {out}")
