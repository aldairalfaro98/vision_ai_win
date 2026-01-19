# app/tests/test_liveness_frame_endpoint.py
import io
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_liveness_frame_endpoint_smoke(monkeypatch):
    import app.application.liveness_service as ls

    class FakeAnti:
        is_real = True
        confidence = 0.99
        bbox = (0.0, 0.0, 100.0, 100.0)

    ears = [0.30, 0.30, 0.30, 0.12, 0.30]
    idx = {"i": 0}

    class FakeLM:
        def __init__(self, ear):
            self.ear_left = ear
            self.ear_right = ear
            self.rvec = np.zeros((3, 1), dtype=float)
            self.tvec = np.zeros((3, 1), dtype=float)

    monkeypatch.setattr(ls._predictor, "predict_from_bgr", lambda _img: FakeAnti())
    monkeypatch.setattr(ls._landmarks, "analyze", lambda _img, _bbox: FakeLM(ears[min(idx["i"], len(ears)-1)]))

    sid = "t1"
    img = b"\xff\xd8\xff"  # bytes dummy; no decodifica porque predictor/landmarks están mocked

    # OJO: como decode_image_bytes_to_bgr sí corre, evitamos decodificar parcheándolo también
    monkeypatch.setattr(ls, "decode_image_bytes_to_bgr", lambda _b: object())

    for _ in range(5):
        idx["i"] += 1
        res = client.post("/liveness/frame", params={"session_id": sid},
                          files={"file": ("f.jpg", io.BytesIO(img), "image/jpeg")})
        assert res.status_code == 200

    data = res.json()
    assert "checks" in data
    assert isinstance(data["checks"]["blink_detected"], bool)
