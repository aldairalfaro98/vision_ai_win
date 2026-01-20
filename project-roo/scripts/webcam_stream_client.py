# scripts/webcam_stream_client.py
import time
import uuid
from dataclasses import dataclass

import cv2
import requests


@dataclass(frozen=True)
class ApiResult:
    live: bool
    conf: float
    blink: bool
    head: bool


def _encode_jpg(frame, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _call_frame(session: requests.Session, url: str, sid: str, jpg: bytes) -> ApiResult:
    files = {"file": ("frame.jpg", jpg, "image/jpeg")}
    resp = session.post(url, params={"session_id": sid}, files=files, timeout=5.0)
    resp.raise_for_status()
    data = resp.json()
    c = data.get("checks") or {}
    return ApiResult(
        live=bool(data["liveness"]),
        conf=float(data["confidence"]),
        blink=bool(c.get("blink_detected", False)),
        head=bool(c.get("head_movement", False)),
    )


def _overlay(frame, r: ApiResult | None, fps: float, api_ms: float | None) -> None:
    w = frame.shape[1]
    color = (0, 255, 0) if (r and r.live) else (0, 0, 255)
    titulo = "VIVO" if (r and r.live) else "NO VIVO"
    conf = f"{r.conf:.2f}" if r else "--"
    blink = f"{r.blink}" if r else "--"
    head = f"{r.head}" if r else "--"
    lat = f"{api_ms:.0f}ms" if api_ms is not None else "--"
    txt = f"conf={conf} | parpadeo={blink} | cabeza={head} | fps={fps:.1f} | api={lat}"
    cv2.rectangle(frame, (0, 0), (w, 120), (10, 10, 10), -1)
    cv2.putText(frame, titulo, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    cv2.putText(frame, txt, (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def run(base_url: str = "http://127.0.0.1:8000", camera_index: int = 0) -> None:
    sid = str(uuid.uuid4())[:8]
    url = f"{base_url.rstrip('/')}/liveness/frame"
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam")

    session = requests.Session()
    last_t = time.time()
    fps = 0.0
    last_api_ms = None
    last_res = None

    print(f"[stream] session_id={sid} -> {url} | q=salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps else (1.0 / dt)

        small = cv2.resize(frame, (640, 360))
        t0 = time.time()
        try:
            jpg = _encode_jpg(small)
            last_res = _call_frame(session, url, sid, jpg)
            last_api_ms = (time.time() - t0) * 1000.0
        except Exception:
            last_res = None
            last_api_ms = None

        _overlay(frame, last_res, fps, last_api_ms)
        cv2.imshow("Liveness demo (stream)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    session.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
