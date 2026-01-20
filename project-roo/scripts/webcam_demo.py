# scripts/webcam_demo.py
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import requests


@dataclass(frozen=True)
class ApiResult:
    liveness: bool
    confidence: float
    blink_detected: bool
    head_movement: bool


def _overlay(
    frame,
    result: ApiResult | None,
    fps: float,
    window_size: int,
    api_latency_ms: float | None,
):
    h, w = frame.shape[:2]
    pad = 16

    if result is None:
        is_live = False
        titulo = "SIN RESULTADO"
        linea = "conf=-- | parpadeo=-- | cabeza=--"
    else:
        is_live = result.liveness
        titulo = "VIVO" if is_live else "NO VIVO"
        linea = (
            f"conf={result.confidence:.2f} | parpadeo={result.blink_detected} "
            f"| cabeza={result.head_movement}"
        )

    color = (0, 255, 0) if is_live else (0, 0, 255)
    latencia = f"{api_latency_ms:.0f}ms" if api_latency_ms is not None else "--"
    linea_meta = f"fps={fps:.1f} | latencia api={latencia}"

    # Fondo para texto (simple, legible)
    cv2.rectangle(frame, (0, 0), (w, 130), (10, 10, 10), thickness=-1)

    cv2.putText(frame, titulo, (pad, 55), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    cv2.putText(frame, linea, (pad, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, linea_meta, (pad, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _encode_jpeg(frame, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _call_api_check_frames(
    session: requests.Session,
    api_url: str,
    frames_jpeg: List[bytes],
    timeout_s: float = 10.0,
) -> ApiResult:
    # Enviamos lista de archivos bajo el campo "files" (como tus tests)
    files = []
    for i, b in enumerate(frames_jpeg):
        files.append(("files", (f"frame_{i}.jpg", b, "image/jpeg")))

    resp = session.post(api_url, files=files, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    checks = data.get("checks") or {}
    return ApiResult(
        liveness=bool(data["liveness"]),
        confidence=float(data["confidence"]),
        blink_detected=bool(checks.get("blink_detected", False)),
        head_movement=bool(checks.get("head_movement", False)),
    )


def run_demo(
    base_url: str = "http://127.0.0.1:8000",
    camera_index: int = 0,
    window_size: int = 8,
    min_interval_s: float = 0.50,
) -> None:
    api_url = f"{base_url.rstrip('/')}/liveness/check-frames"
    session = requests.Session()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1 or check permissions.")

    last_call = 0.0
    frames: List[bytes] = []

    last_t = time.time()
    fps = 0.0

    last_result: ApiResult | None = None
    last_api_ms: float | None = None

    print(f"[demo] Sending batches to: {api_url}")
    print("[demo] Keys: q=quit | space=force send batch")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_small = cv2.resize(frame, (640, 360))

        # FPS (solo visual)
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # acumulamos frames para el batch
        try:
            frames.append(_encode_jpeg(frame_small, quality=65))
        except Exception:
            # si fallara el encode, no detenemos el loop
            frames = []
            continue

        # decide si mandamos
        force_send = (cv2.waitKey(1) & 0xFF) == ord(" ")
        should_send = (len(frames) >= window_size) and ((now - last_call) >= min_interval_s)

        if force_send or should_send:
            batch = frames[-window_size:] if len(frames) >= window_size else frames[:]
            frames = []  # reset simple

            last_call = now
            t0 = time.time()
            try:
                last_result = _call_api_check_frames(session, api_url, batch)
                last_api_ms = (time.time() - t0) * 1000.0
            except requests.RequestException as e:
                last_result = None
                last_api_ms = None
                print(f"[demo] API error: {e}")

        _overlay(frame, last_result, fps=fps, window_size=window_size, api_latency_ms=last_api_ms)
        cv2.imshow("Liveness demo (client)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    session.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
