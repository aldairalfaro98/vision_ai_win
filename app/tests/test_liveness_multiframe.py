from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_liveness_accepts_multiple_files_and_keeps_contract():
    img_path = Path(__file__).parent / "assets" / "face.jpeg"

    with img_path.open("rb") as f1, img_path.open("rb") as f2, img_path.open("rb") as f3:
        files = [
            ("files", ("face1.jpeg", f1, "image/jpeg")),
            ("files", ("face2.jpeg", f2, "image/jpeg")),
            ("files", ("face3.jpeg", f3, "image/jpeg")),
        ]
        res = client.post("/liveness/check-frames", files=files)

    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"liveness", "confidence", "checks"}
    assert set(data["checks"].keys()) == {"blink_detected", "head_movement"}
    assert 0.0 <= float(data["confidence"]) <= 1.0
    assert isinstance(data["liveness"], bool)
