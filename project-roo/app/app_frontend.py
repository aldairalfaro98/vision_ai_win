# project-roo/app/app_frontend.py
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st


@dataclass(frozen=True)
class ApiResult:
    liveness: bool
    confidence: float
    anti_label: str
    blink_count: int
    head_movement: bool


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam. Prueba camera_index=1 o revisa permisos.")
    return cap


def _countdown(seconds: int) -> None:
    ph = st.empty()
    for s in range(int(seconds), 0, -1):
        ph.markdown(f"### Preparación... **{s}**")
        time.sleep(1)
    ph.empty()


def _record_clip(
    cap: cv2.VideoCapture,
    *,
    duration_s: float,
    target_fps: int,
    mirror: bool,
    preview_box,
) -> Tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []

    start = time.perf_counter()
    interval = 1.0 / float(target_fps)
    next_t = start

    while True:
        now = time.perf_counter()
        if (now - start) >= float(duration_s):
            break

        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        frames.append(frame)
        preview_box.image(_bgr_to_rgb(frame), channels="RGB", use_container_width=True)

        next_t += interval
        sleep_s = max(0.0, next_t - time.perf_counter())
        if sleep_s > 0:
            time.sleep(sleep_s)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return frames, float(elapsed_ms)


def _frames_to_avi_bytes(frames: list[np.ndarray], *, fps: int) -> bytes:
    if not frames:
        return b""

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    fd, path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)  # Importante en Windows: libera el handle

    try:
        writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        try:
            for f in frames:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h))
                writer.write(f)
        finally:
            writer.release()

        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _call_api_video(api_url: str, video_bytes: bytes, timeout_s: float = 30.0) -> dict:
    files = {"file": ("clip.avi", video_bytes, "video/x-msvideo")}
    resp = requests.post(api_url, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _call_api_frames(api_url: str, frames: list[np.ndarray], timeout_s: float = 30.0) -> dict:
    files = []
    for i, f in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        files.append(("files", (f"f{i}.jpg", buf.tobytes(), "image/jpeg")))

    resp = requests.post(api_url, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _parse_result(payload: dict) -> ApiResult:
    checks = payload.get("checks") or {}
    anti = checks.get("anti_spoof") or {}

    return ApiResult(
        liveness=bool(payload.get("liveness")),
        confidence=float(payload.get("confidence") or 0.0),
        anti_label=str(anti.get("label") or "UNKNOWN"),
        blink_count=int(checks.get("blink_count") or 0),
        head_movement=bool(checks.get("head_movement", False)),
    )


def _render_status(result: Optional[ApiResult]) -> None:
    if result is None:
        st.info("Sin resultado todavía.")
        return

    if result.liveness:
        st.success(f"✅ VIVO | conf={result.confidence:.2f} | PAD={result.anti_label}")
    else:
        st.error(f"⛔ NO VIVO | conf={result.confidence:.2f} | PAD={result.anti_label}")

    actual_output = {
        "parpadeos_detectados": result.blink_count,
        "mov_cabeza": result.head_movement,
    }

    expected_output = {
        "liveness": bool(result.liveness),
        "confidence": float(round(result.confidence, 2)),
        "checks": {
            "blink_detected": bool(result.blink_count > 0),   # regla UI: >0 => True
            "head_movement": bool(result.head_movement),
            "pad_result": str(result.anti_label),             # REAL / FAKE
        },
    }

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.caption("Output actual (debug)")
        st.json(actual_output)

    with col_b:
        st.caption("Output esperado (prueba técnica)")
        st.json(expected_output)


def main() -> None:
    st.set_page_config(page_title="Liveness (Demo)", layout="wide")
    st.title("Liveness Detection — Demo híbrida (2s)")

    st.info("Instrucción: **Parpadea 2 veces** y **mueve la cabeza izquierda↔derecha** durante el clip.")

    with st.sidebar:
        base_url = st.text_input("Backend URL", "http://127.0.0.1:8000")
        camera_index = st.number_input("camera_index", min_value=0, max_value=5, value=0, step=1)
        mirror = st.checkbox("Espejo (UX)", value=True)

        duration_s = st.selectbox("Duración clip", options=[5.0], index=0)
        target_fps = st.selectbox("FPS", options=[30], index=0)

        api_video = f"{base_url.rstrip('/')}/liveness/check-video"
        api_frames = f"{base_url.rstrip('/')}/liveness/check-frames"

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.subheader("Vista local")
        preview = st.empty()

    with col2:
        st.subheader("Resultado servidor")
        status_box = st.empty()
        st.caption("Primero intenta /check-video; si falla, cae a /check-frames.")

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    start = st.button("🎬 Iniciar prueba", type="primary")
    if not start:
        _render_status(st.session_state["last_result"])
        return

    cap = _open_camera(int(camera_index))
    try:
        _countdown(3)

        t0 = time.perf_counter()
        frames, rec_ms = _record_clip(
            cap,
            duration_s=float(duration_s),
            target_fps=int(target_fps),
            mirror=bool(mirror),
            preview_box=preview,
        )

        with status_box:
            st.write(f"📦 Frames capturados: {len(frames)} | tiempo captura: {rec_ms:.0f}ms")
            st.write("⏫ Enviando al backend...")

        payload = None
        try:
            video_bytes = _frames_to_avi_bytes(frames, fps=int(target_fps))
            with status_box:
                st.write(f"📦 AVI bytes: {len(video_bytes)}")

            if len(video_bytes) > 0:
                payload = _call_api_video(api_video, video_bytes)
            else:
                payload = None  # cae a /check-frames
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) not in (404, 415):
                raise
        except Exception:
            payload = None

        if payload is None:
            payload = _call_api_frames(api_frames, frames)

        result = _parse_result(payload)
        st.session_state["last_result"] = result

        t1 = (time.perf_counter() - t0) * 1000.0
        with status_box:
            st.write(f"⏱️ Total (captura+backend): {t1:.0f}ms")
        _render_status(result)

    except requests.RequestException as e:
        st.session_state["last_result"] = None
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                detail = f" | body={e.response.text}"
            except Exception:
                detail = ""
        st.error(f"Error llamando al backend: {e}{detail}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
