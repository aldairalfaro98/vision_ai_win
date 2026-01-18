# app/tests/test_liveness_contract.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_liveness_contract_shape():
    res = client.post("/liveness/check")
    assert res.status_code == 200

    data = res.json()
    assert set(data.keys()) == {"liveness", "confidence", "checks"}
    assert set(data["checks"].keys()) == {"blink_detected", "head_movement"}

    assert isinstance(data["liveness"], bool)
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= float(data["confidence"]) <= 1.0
    assert isinstance(data["checks"]["blink_detected"], bool)
    assert isinstance(data["checks"]["head_movement"], bool)
