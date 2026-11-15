"""Feature engineering + recursive multi-day forecasting helpers.

All features are derived from the Close price only, so the same function
can be re-applied on predicted values to forecast several days ahead.
"""

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4, 5, 6, 7]
MIN_HISTORY = 31  # 30-day moving average + 1 day for returns


def build_features(close: pd.Series) -> pd.DataFrame:
    """Create lag, moving-average and volatility features from closes."""
    df = pd.DataFrame({"close": close.astype(float)})
    for lag in LAGS:
        df[f"lag_{lag}"] = df["close"].shift(lag - 1)
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_30"] = df["close"].rolling(30).mean()
    df["volatility_7"] = df["close"].pct_change().rolling(7).std()
    df["return_1"] = df["close"].pct_change()
    return df


FEATURE_COLS = [f"lag_{lag}" for lag in LAGS] + [
    "ma_7",
    "ma_30",
    "volatility_7",
    "return_1",
]


def make_supervised(close: pd.Series):
    """Return X (features at day t) and y (close at day t+1)."""
    df = build_features(close)
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    return df[FEATURE_COLS], df["target"]


def recursive_forecast(model, history, n_days: int = 7) -> list:
    """Forecast n_days ahead, feeding each prediction back as input."""
    values = [float(v) for v in history]
    if len(values) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} historical closes")
    preds = []
    for _ in range(n_days):
        feats = build_features(pd.Series(values[-60:]))[FEATURE_COLS]
        next_val = float(model.predict(feats.iloc[[-1]])[0])
        preds.append(next_val)
        values.append(next_val)
    return preds


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 3)}


def clean_and_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data and add technical indicators.

    Keeps Date + Close (needed by training) and adds 7/30-day moving
    averages, 7/30-day volatility and daily returns for analysis.
    """
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close"]).drop_duplicates(subset="Date").sort_values("Date")
    df = df[df["Close"] > 0]
    feats = build_features(df["Close"].reset_index(drop=True))
    out = pd.DataFrame(
        {
            "Date": df["Date"].dt.date.values,
            "Close": df["Close"].values,
            "Volume": df["Volume"].values if "Volume" in df else np.nan,
            "ma_7": feats["ma_7"].values,
            "ma_30": feats["ma_30"].values,
            "volatility_7": feats["volatility_7"].values,
            "volatility_30": df["Close"].pct_change().rolling(30).std().values,
            "return_1": feats["return_1"].values,
        }
    )
    return out.reset_index(drop=True)
