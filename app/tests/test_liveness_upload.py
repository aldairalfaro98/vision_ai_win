from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_liveness_accepts_file_upload_and_keeps_contract():
    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    with img_path.open("rb") as f:
        files = {"file": ("face.jpeg", f, "image/jpeg")}
        res = client.post("/liveness/check", files=files)

    assert res.status_code == 200
    data = res.json()

    assert set(data.keys()) == {"liveness", "confidence", "checks"}
    assert set(data["checks"].keys()) == {"blink_detected", "head_movement"}
    assert 0.0 <= float(data["confidence"]) <= 1.0
    assert isinstance(data["liveness"], bool)
