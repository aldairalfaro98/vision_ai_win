# app/tests/test_liveness_landmarks_logs.py
from pathlib import Path
import logging

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_multiframe_emits_landmarks_logs(caplog):
    caplog.set_level(logging.INFO, logger="liveness")

    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    with img_path.open("rb") as f1, img_path.open("rb") as f2, img_path.open("rb") as f3:
        files = [
            ("files", ("face1.jpeg", f1, "image/jpeg")),
            ("files", ("face2.jpeg", f2, "image/jpeg")),
            ("files", ("face3.jpeg", f3, "image/jpeg")),
        ]
        res = client.post("/liveness/check-frames", files=files)

    assert res.status_code == 200

    messages = [rec.getMessage() for rec in caplog.records]

    # Buscamos evidencia de que se intentó landmarks/pose
    assert any("ear_left=" in m for m in messages) or any("landmarks_or_pose_failed" in m for m in messages)
