import pytest
from fastapi.testclient import TestClient
from api.app import app, load_services


@pytest.fixture(scope="module")
def client():
    load_services()
    return TestClient(app)


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["pipeline_loaded"] is True
    assert data["models_loaded"] is True


def test_api_realtime_score(client):
    payload = {
        "user_id": "USR_00010",
        "card_id": "CARD_00010",
        "amount": 9900.0,
        "channel": "wire_transfer",
        "mcc": "6051",
        "device_id": "DEV_TEST_ALERT",
        "ip_address": "198.51.100.99",
    }
    res = client.post("/api/v1/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "composite_risk_score" in data
    assert "risk_tier" in data
    assert "case_id" in data
    assert data["processing_time_ms"] > 0


def test_api_metrics(client):
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_cases" in data
    assert "open_cases" in data
