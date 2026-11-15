"""Flask REST API serving the 7-day BTC close-price forecaster.

Endpoints
  GET  /health   -> status, uptime, loaded model version
  POST /predict  -> 7-day (or N-day) forecast with dates
  GET  /metrics  -> latest validation metrics (MAE, RMSE, MAPE)
  POST /reload   -> re-read model weights from disk (called by Airflow)

Production:  gunicorn --bind 0.0.0.0:5000 app:app
"""

import json
import logging
import os
import threading
import time
from datetime import date, timedelta

import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.config import METRICS_PATH, MODEL_PATH
from src.features import MIN_HISTORY, recursive_forecast

MAX_FORECAST_DAYS = 30

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("btc-api")


class ModelStore:
    """Holds the loaded model and reloads it when the file on disk changes."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.artifact = None
        self.mtime = None
        self.lock = threading.Lock()

    def load(self) -> bool:
        with self.lock:
            try:
                self.artifact = joblib.load(self.model_path)
                self.mtime = os.path.getmtime(self.model_path)
                log.info("Loaded model %s from %s", self.version, self.model_path)
                return True
            except FileNotFoundError:
                log.warning("Model file not found: %s", self.model_path)
            except Exception:  # noqa: BLE001
                log.exception("Failed to load model from %s", self.model_path)
            self.artifact, self.mtime = None, None
            return False

    def refresh_if_changed(self):
        """Pick up a newly trained model (e.g. written by Airflow to the PVC)."""
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            return
        if mtime != self.mtime:
            self.load()

    @property
    def loaded(self) -> bool:
        return self.artifact is not None

    @property
    def version(self):
        return self.artifact["version"] if self.artifact else None


def bad_request(message: str):
    return jsonify({"error": "Bad Request", "message": message}), 400


def parse_predict_payload(payload):
    """Validate the /predict body. Returns (forecast_days, history) or raises ValueError."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    days = payload.get("forecast_days", 7)
    if isinstance(days, bool) or not isinstance(days, int):
        raise ValueError("'forecast_days' must be an integer")
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"'forecast_days' must be between 1 and {MAX_FORECAST_DAYS}")

    history = payload.get("history")
    if history is not None:
        if not isinstance(history, list) or len(history) < MIN_HISTORY:
            raise ValueError(f"'history' must be a list of at least {MIN_HISTORY} closing prices")
        for v in history:
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError("'history' values must be positive numbers (no nulls)")
    return days, history


def create_app(model_path=MODEL_PATH, metrics_path=METRICS_PATH) -> Flask:
    app = Flask(__name__)
    store = ModelStore(model_path)
    store.load()
    started = time.time()
    app.config["MODEL_STORE"] = store

    @app.get("/health")
    def health():
        store.refresh_if_changed()
        body = {
            "status": "ok" if store.loaded else "model_not_loaded",
            "uptime_seconds": round(time.time() - started, 1),
            "model_version": store.version,
            "model_stage": os.getenv("MODEL_STAGE", "production"),
            "pod": os.getenv("HOSTNAME", "local"),
        }
        return jsonify(body), (200 if store.loaded else 503)

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json(silent=False) if request.data else None
        except Exception:  # noqa: BLE001  (malformed JSON)
            return bad_request("Malformed JSON body")
        try:
            days, history = parse_predict_payload(payload)
        except ValueError as exc:
            return bad_request(str(exc))

        store.refresh_if_changed()
        if not store.loaded:
            return jsonify({"error": "Service Unavailable", "message": "Model not loaded"}), 503

        art = store.artifact
        if history is None:
            history = art["history"]
            start = date.fromisoformat(art["last_date"]) + timedelta(days=1)
            source = "stored_history"
        else:
            start = date.today() + timedelta(days=1)
            source = "custom_history"

        try:
            preds = recursive_forecast(art["model"], history, days)
        except Exception as exc:  # noqa: BLE001
            log.exception("Prediction failed")
            return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

        return jsonify(
            {
                "model_version": store.version,
                "forecast_days": days,
                "input_source": source,
                "predictions": [
                    {"date": str(start + timedelta(days=i)), "predicted_close": round(p, 2)}
                    for i, p in enumerate(preds)
                ],
            }
        )

    @app.get("/metrics")
    def metrics():
        try:
            with open(metrics_path) as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify({"error": "Not Found", "message": "No metrics available yet"}), 404
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not read metrics")
            return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

    @app.post("/reload")
    def reload_model():
        api_key = os.getenv("API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401
        ok = store.load()
        status = 200 if ok else 500
        return jsonify({"reloaded": ok, "model_version": store.version}), status

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return jsonify({"error": exc.name, "message": exc.description}), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        log.exception("Unhandled error")
        return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
