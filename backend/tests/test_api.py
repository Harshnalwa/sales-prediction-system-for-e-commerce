from fastapi import HTTPException

import main


VALID_PAYLOAD = {
    "Store": 1,
    "Dept": 1,
    "IsHoliday": False,
    "Temperature": 42.31,
    "Fuel_Price": 2.572,
    "MarkDown1": 0,
    "MarkDown2": 0,
    "MarkDown3": 0,
    "MarkDown4": 0,
    "MarkDown5": 0,
    "CPI": 211.096358,
    "Unemployment": 8.106,
    "Type": "A",
    "Size": 151315,
    "Year": 2010,
    "Month": 2,
    "Week": 5,
}


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Sales Prediction API is running"


def test_health_endpoint_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_metrics_endpoint(client):
    response = client.get("/model-metrics")
    assert response.status_code == 200
    assert response.json()["r2_score"] == 0.91
    assert response.json()["mae"] == 1200.0


def test_prediction_endpoint(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {
        "predicted_weekly_sales": 12345.68,
        "store": 1,
        "department": 1,
    }


def test_prediction_rejects_invalid_store_id(client):
    payload = {**VALID_PAYLOAD, "Store": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_rejects_invalid_store_type(client):
    payload = {**VALID_PAYLOAD, "Type": "D"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_returns_503_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(main, "load_model_bundle", lambda: None)

    def missing_model():
        raise HTTPException(status_code=503, detail="Model is not available.")

    main.app.dependency_overrides[main.get_model_bundle] = missing_model
    try:
        from fastapi.testclient import TestClient

        with TestClient(main.app) as test_client:
            response = test_client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    finally:
        main.app.dependency_overrides.clear()


def test_health_endpoint_stays_available_without_model(monkeypatch):
    monkeypatch.setattr(main, "load_model_bundle", lambda: None)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}


def test_metrics_endpoint_has_safe_fallback_without_model(monkeypatch):
    monkeypatch.setattr(main, "load_model_bundle", lambda: None)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        response = test_client.get("/model-metrics")

    assert response.status_code == 200
    assert response.json()["r2_score"] is None
    assert "message" in response.json()
