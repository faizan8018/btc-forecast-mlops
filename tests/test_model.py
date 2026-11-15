"""Unit tests for feature engineering and the forecasting helpers."""

import pytest

from src.features import FEATURE_COLS, make_supervised, recursive_forecast, regression_metrics
from src.train_forecast import new_model


def test_make_supervised_shapes(synthetic_close):
    X, y = make_supervised(synthetic_close)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y)
    assert not X.isna().any().any()


def test_recursive_forecast_length(synthetic_close):
    X, y = make_supervised(synthetic_close)
    model = new_model().fit(X, y)
    preds = recursive_forecast(model, synthetic_close.values, 7)
    assert len(preds) == 7


def test_recursive_forecast_needs_enough_history(synthetic_close):
    X, y = make_supervised(synthetic_close)
    model = new_model().fit(X, y)
    with pytest.raises(ValueError):
        recursive_forecast(model, [100.0] * 10, 7)


def test_regression_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m == {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
