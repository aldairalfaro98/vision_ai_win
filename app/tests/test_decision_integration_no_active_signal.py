# app/tests/test_decision_integration_no_active_signal.py
import io
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_liveness_false_when_no_active_signal(monkeypatch):
    import app.application.liveness_service as ls

    # 1) Forzar anti-spoof a "pasar" siempre
    class FakeAnti:
        def __init__(self):
            self.is_real = True
            self.confidence = 0.99
            self.bbox = (0.0, 0.0, 100.0, 100.0)

    def fake_predict(_img):
        return FakeAnti()

    monkeypatch.setattr(ls._predictor, "predict_from_bgr", fake_predict)

    # 2) Forzar landmarks sin blink (EAR estable) y sin movimiento (rvec estable)
    class FakeLM:
        def __init__(self):
            self.ear_left = 0.30
            self.ear_right = 0.30
            self.rvec = np.zeros((3, 1), dtype=float)
            self.tvec = np.zeros((3, 1), dtype=float)

    def fake_analyze(_img, _bbox):
        return FakeLM()

    monkeypatch.setattr(ls._landmarks, "analyze", fake_analyze)

    # 3) Enviar frames suficientes
    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    img_bytes = img_path.read_bytes()
    files = [("files", (f"f{i}.jpeg", io.BytesIO(img_bytes), "image/jpeg")) for i in range(6)]

    res = client.post("/liveness/check-frames", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["checks"]["blink_detected"] is False
    assert data["checks"]["head_movement"] is False
    assert data["liveness"] is False  # <- regla híbrida
