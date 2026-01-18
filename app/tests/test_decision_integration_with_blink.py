# app/tests/test_decision_integration_with_blink.py
import io
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_liveness_true_when_blink_present(monkeypatch):
    import app.application.liveness_service as ls

    # 1) Forzar anti-spoof a "pasar" siempre
    class FakeAnti:
        def __init__(self):
            self.is_real = True
            self.confidence = 0.99
            self.bbox = (0.0, 0.0, 100.0, 100.0)

    monkeypatch.setattr(ls._predictor, "predict_from_bgr", lambda _img: FakeAnti())

    # 2) Simular EAR con blink (abierto->cerrado->abierto) y rvec estable
    ears = [(0.30, 0.30), (0.29, 0.29), (0.30, 0.30), (0.12, 0.12), (0.11, 0.11), (0.29, 0.29)]
    idx = {"i": 0}

    class FakeLM:
        def __init__(self, el, er):
            self.ear_left = el
            self.ear_right = er
            self.rvec = np.zeros((3, 1), dtype=float)
            self.tvec = np.zeros((3, 1), dtype=float)

    def fake_analyze(_img, _bbox):
        i = idx["i"]
        idx["i"] += 1
        el, er = ears[min(i, len(ears) - 1)]
        return FakeLM(el, er)

    monkeypatch.setattr(ls._landmarks, "analyze", fake_analyze)

    # 3) Enviar 6 frames
    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    img_bytes = img_path.read_bytes()
    files = [("files", (f"f{i}.jpeg", io.BytesIO(img_bytes), "image/jpeg")) for i in range(6)]

    res = client.post("/liveness/check-frames", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["checks"]["blink_detected"] is True
    assert data["liveness"] is True  # <- regla híbrida
