# scripts/webcam_demo_local.py
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2

from uniface import RetinaFace
from uniface.spoofing import MiniFASNet

from app.infrastructure.uniface.landmarks import UniFaceLandmarks
from app.infrastructure.uniface.pose import rvec_to_euler_degrees
from app.domain.blink import detect_blink_from_ear_series
from app.domain.head_movement import detect_head_movement_from_angles
from app.domain.decision import decide_liveness


@dataclass(frozen=True)
class FrameSignals:
    is_real: bool
    confidence: float
    ear_avg: float | None
    angles: Tuple[float, float, float] | None


def _draw_overlay(frame, live: bool, conf: float, blink: bool, head: bool, fps: float) -> None:
    h, w = frame.shape[:2]
    color = (0, 255, 0) if live else (0, 0, 255)
    titulo = "VIVO" if live else "NO VIVO"
    linea = f"conf={conf:.2f} | parpadeo={blink} | cabeza={head} | fps={fps:.1f}"
    cv2.rectangle(frame, (0, 0), (w, 90), (10, 10, 10), thickness=-1)
    cv2.putText(frame, titulo, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    cv2.putText(frame, linea, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _compute_signals(
    frame_bgr,
    detector: RetinaFace,
    spoofer: MiniFASNet,
    landmarks: UniFaceLandmarks,
) -> FrameSignals | None:
    faces = detector.detect(frame_bgr)
    if not faces:
        return None

    face = faces[0]
    res = spoofer.predict(frame_bgr, face.bbox)

    ear_avg = None
    angles = None
    lm = landmarks.analyze(frame_bgr, face.bbox)
    if lm:
        ear_avg = float((lm.ear_left + lm.ear_right) / 2.0)
        angles = rvec_to_euler_degrees(lm.rvec)

    return FrameSignals(
        is_real=bool(res.is_real),
        confidence=float(res.confidence),
        ear_avg=ear_avg,
        angles=angles,
    )


def run(camera_index: int = 0, window_size: int = 6) -> None:
    detector = RetinaFace()
    spoofer = MiniFASNet()
    landmarks = UniFaceLandmarks()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam. Prueba camera_index=1 o permisos.")

    ears: List[float] = []
    angs: List[Tuple[float, float, float]] = []

    last_t = time.time()
    fps = 0.0

    print("[local demo] q=salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # espejo (UX)

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)

        sig = _compute_signals(frame, detector, spoofer, landmarks)
        if sig is None:
            _draw_overlay(frame, live=False, conf=0.0, blink=False, head=False, fps=fps)
            cv2.imshow("Liveness demo (local)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue

        if sig.ear_avg is not None:
            ears.append(sig.ear_avg)
            ears[:] = ears[-window_size:]

        if sig.angles is not None:
            angs.append(sig.angles)
            angs[:] = angs[-window_size:]

        blink = detect_blink_from_ear_series(ears)
        head = detect_head_movement_from_angles(angs)
        anti_ok = bool(sig.is_real)  # aquí “imitamos” a la doc: el modelo manda
        live = decide_liveness(anti_spoof_passed=anti_ok, blink_detected=blink, head_movement=head)

        _draw_overlay(frame, live=live, conf=sig.confidence, blink=blink, head=head, fps=fps)
        cv2.imshow("Liveness demo (local)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
