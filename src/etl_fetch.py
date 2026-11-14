"""Fetch daily BTC-USD OHLCV data and store it in raw storage (CSV).

Primary source: yfinance. Fallback: Binance public klines API.
Usage:  python -m src.etl_fetch
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import RAW_DATA_PATH, START_DATE, TICKER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("etl_fetch")

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_yfinance(start: str = START_DATE) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(TICKER, start=start, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned no rows")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance versions
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[COLUMNS]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def fetch_binance(start: str = START_DATE) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    while True:
        params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = COLUMNS
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.date
    df[COLUMNS[1:]] = df[COLUMNS[1:]].astype(float)
    return df


def fetch_btc_data(start: str = START_DATE) -> pd.DataFrame:
    try:
        df = fetch_yfinance(start)
        log.info("Fetched %d rows from yfinance", len(df))
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance failed (%s) - falling back to Binance", exc)
        df = fetch_binance(start)
        log.info("Fetched %d rows from Binance", len(df))
    df = df.dropna().drop_duplicates(subset="Date").sort_values("Date")
    return df.reset_index(drop=True)


def save_raw(df: pd.DataFrame, path=RAW_DATA_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(
        "Saved raw data -> %s (%d rows, %s to %s)",
        path,
        len(df),
        df["Date"].iloc[0],
        df["Date"].iloc[-1],
    )
    return str(path)


if __name__ == "__main__":
    save_raw(fetch_btc_data())
