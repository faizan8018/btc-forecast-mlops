"""Shared fixtures: a small model trained on synthetic prices (no internet needed)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app.app import create_app
from src.features import FEATURE_COLS, make_supervised
from src.train_forecast import new_model


@pytest.fixture(scope="session")
def synthetic_close():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    prices = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    return pd.Series(prices, index=dates)


@pytest.fixture()
def model_files(tmp_path, synthetic_close):
    X, y = make_supervised(synthetic_close)
    artifact = {
        "model": new_model().fit(X, y),
        "version": "vTEST",
        "feature_cols": FEATURE_COLS,
        "last_date": str(synthetic_close.index[-1].date()),
        "history": synthetic_close.values[-60:].tolist(),
    }
    model_path, metrics_path = tmp_path / "model.pkl", tmp_path / "metrics.json"
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps({"MAE": 1.0, "RMSE": 2.0, "MAPE": 3.0}))
    return model_path, metrics_path


@pytest.fixture()
def client(model_files):
    app = create_app(*model_files)
    app.testing = True
    return app.test_client()


@pytest.fixture()
def client_no_model(tmp_path):
    app = create_app(tmp_path / "missing.pkl", tmp_path / "missing.json")
    app.testing = True
    return app.test_client()
