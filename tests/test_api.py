"""Unit tests for the Flask API endpoints."""


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_version"] == "vTEST"
    assert "uptime_seconds" in body


def test_health_without_model_returns_503(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 503
    assert r.get_json()["status"] == "model_not_loaded"


def test_predict_default_7_days(client):
    r = client.post("/predict")
    assert r.status_code == 200
    body = r.get_json()
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    assert {"date", "predicted_close"} <= body["predictions"][0].keys()
    assert all(p["predicted_close"] > 0 for p in body["predictions"])


def test_predict_custom_days(client):
    r = client.post("/predict", json={"forecast_days": 3})
    assert r.status_code == 200
    assert len(r.get_json()["predictions"]) == 3


def test_predict_custom_history(client, synthetic_close):
    history = synthetic_close.values[-40:].tolist()
    r = client.post("/predict", json={"history": history})
    assert r.status_code == 200
    assert r.get_json()["input_source"] == "custom_history"


def test_predict_invalid_days_returns_400(client):
    for bad in [0, 31, "seven", 2.5, True]:
        r = client.post("/predict", json={"forecast_days": bad})
        assert r.status_code == 400, bad


def test_predict_short_history_returns_400(client):
    r = client.post("/predict", json={"history": [100.0] * 5})
    assert r.status_code == 400


def test_predict_null_in_history_returns_400(client):
    r = client.post("/predict", json={"history": [100.0] * 40 + [None]})
    assert r.status_code == 400


def test_predict_malformed_json_returns_400(client):
    r = client.post("/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400


def test_predict_non_object_body_returns_400(client):
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 400


def test_predict_without_model_returns_503(client_no_model):
    assert client_no_model.post("/predict").status_code == 503


def test_metrics_ok(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert {"MAE", "RMSE", "MAPE"} <= r.get_json().keys()


def test_metrics_missing_returns_404(client_no_model):
    assert client_no_model.get("/metrics").status_code == 404


def test_reload(client):
    r = client.post("/reload")
    assert r.status_code == 200
    assert r.get_json()["reloaded"] is True


def test_reload_requires_api_key_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    assert client.post("/reload").status_code == 401
    r = client.post("/reload", headers={"X-API-Key": "secret123"})
    assert r.status_code == 200


def test_unknown_route_returns_json_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert r.get_json()["error"] == "Not Found"


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_index_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "POST /predict" in r.get_json()["endpoints"]
