# app/application/liveness_service.py
from __future__ import annotations

import logging
from statistics import median
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.domain.blink import count_blinks_from_ear_series
from app.domain.decision import LivenessSignals, build_decision_strategy, decide_liveness
from app.domain.head_movement import detect_head_movement_from_angles, yaw_range_from_angles
from app.domain.models import AntiSpoofCheck, LivenessChecks, LivenessResponse
from app.infrastructure.uniface.anti_spoof import UniFaceAntiSpoofPredictor, decode_image_bytes_to_bgr
from app.infrastructure.uniface.landmarks import UniFaceLandmarks
from app.infrastructure.uniface.pose import rvec_to_euler_degrees
from app.infrastructure.video_frames import extract_evenly_spaced_frames_bgr

logger = logging.getLogger(__name__)

_anti = UniFaceAntiSpoofPredictor()
_landmarks = UniFaceLandmarks()


class LivenessService:
    def check(self, image_bytes: Optional[bytes]) -> LivenessResponse:
        if not image_bytes:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(),
                message="No image provided.",
            )

        img = decode_image_bytes_to_bgr(image_bytes)
        img = _resize_max_side(img, max_side_px=640)

        anti_res = _anti.predict_from_bgr(img)
        if anti_res is None:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(
                    anti_spoof=AntiSpoofCheck(label="NO_FACE", confidence=0.0, passed=False),
                ),
                message="No face detected.",
            )

        label = "REAL" if anti_res.is_real else "FAKE"
        checks = LivenessChecks(
            anti_spoof=AntiSpoofCheck(label=label, confidence=anti_res.confidence, passed=anti_res.is_real),
        )

        return LivenessResponse(
            liveness=bool(anti_res.is_real),
            confidence=float(anti_res.confidence),
            checks=checks,
            message="Single-frame PAD only (demo helper).",
        )

    def check_video(
        self,
        video_bytes: bytes,
        *,
        max_frames: int,
        preproc_max_side_px: int,
        anti_min_real_frames: int,
        anti_min_confidence: float,
        blink_baseline_frames: int,
        blink_close_ratio: float,
        blink_open_ratio: float,
        blink_min_closed_frames: int,
        blink_min_count: int,
        head_baseline_frames: int,
        head_yaw_delta_deg: float,
        decision_mode: str,
    ) -> LivenessResponse:
        frames_bgr = extract_evenly_spaced_frames_bgr(video_bytes, max_frames=int(max_frames))
        if not frames_bgr:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(),
                message="Could not decode video or extracted 0 frames.",
            )

        return self._evaluate_frames_bgr(
            frames_bgr,
            preproc_max_side_px=preproc_max_side_px,
            anti_min_real_frames=anti_min_real_frames,
            anti_min_confidence=anti_min_confidence,
            blink_baseline_frames=blink_baseline_frames,
            blink_close_ratio=blink_close_ratio,
            blink_open_ratio=blink_open_ratio,
            blink_min_closed_frames=blink_min_closed_frames,
            blink_min_count=blink_min_count,
            head_baseline_frames=head_baseline_frames,
            head_yaw_delta_deg=head_yaw_delta_deg,
            decision_mode=decision_mode,
        )

    def check_frames(
        self,
        frames_bytes: Sequence[bytes],
        *,
        preproc_max_side_px: int,
        anti_min_real_frames: int,
        anti_min_confidence: float,
        blink_baseline_frames: int,
        blink_close_ratio: float,
        blink_open_ratio: float,
        blink_min_closed_frames: int,
        blink_min_count: int,
        head_baseline_frames: int,
        head_yaw_delta_deg: float,
        decision_mode: str,
    ) -> LivenessResponse:
        frames_bgr: List[np.ndarray] = []
        for b in frames_bytes:
            try:
                img = decode_image_bytes_to_bgr(b)
                frames_bgr.append(img)
            except Exception:
                continue

        if not frames_bgr:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(),
                message="No valid frames decoded.",
            )

        return self._evaluate_frames_bgr(
            frames_bgr,
            preproc_max_side_px=preproc_max_side_px,
            anti_min_real_frames=anti_min_real_frames,
            anti_min_confidence=anti_min_confidence,
            blink_baseline_frames=blink_baseline_frames,
            blink_close_ratio=blink_close_ratio,
            blink_open_ratio=blink_open_ratio,
            blink_min_closed_frames=blink_min_closed_frames,
            blink_min_count=blink_min_count,
            head_baseline_frames=head_baseline_frames,
            head_yaw_delta_deg=head_yaw_delta_deg,
            decision_mode=decision_mode,
        )

    def _evaluate_frames_bgr(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        preproc_max_side_px: int,
        anti_min_real_frames: int,
        anti_min_confidence: float,
        blink_baseline_frames: int,
        blink_close_ratio: float,
        blink_open_ratio: float,
        blink_min_closed_frames: int,
        blink_min_count: int,
        head_baseline_frames: int,
        head_yaw_delta_deg: float,
        decision_mode: str,
    ) -> LivenessResponse:
        anti_is_real: List[bool] = []
        anti_confs: List[float] = []
        ear_series: List[float] = []
        angles_series: List[Tuple[float, float, float]] = []
        frames_with_face = 0

        for raw in frames_bgr:
            img = _resize_max_side(raw, max_side_px=int(preproc_max_side_px))

            anti = _anti.predict_from_bgr(img)
            if anti is None:
                continue

            frames_with_face += 1
            anti_is_real.append(bool(anti.is_real))
            anti_confs.append(float(anti.confidence))

            lm = _landmarks.analyze(img, anti.bbox)
            if lm is None:
                continue

            ear_series.append(float((lm.ear_left + lm.ear_right) / 2.0))
            angles_series.append(rvec_to_euler_degrees(lm.rvec))

        if frames_with_face == 0:
            checks = LivenessChecks(
                anti_spoof=AntiSpoofCheck(label="NO_FACE", confidence=0.0, passed=False),
            )
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=checks,
                message="No face detected in any frame.",
            )

        conf_med = float(median(anti_confs)) if anti_confs else 0.0
        real_count = int(sum(1 for x in anti_is_real if x))
        anti_passed = bool(real_count >= int(anti_min_real_frames) and conf_med >= float(anti_min_confidence))

        blink_count = count_blinks_from_ear_series(
            ear_series,
            baseline_frames=int(blink_baseline_frames),
            close_ratio=float(blink_close_ratio),
            open_ratio=float(blink_open_ratio),
            min_closed_frames=int(blink_min_closed_frames),
        )

        head_movement = detect_head_movement_from_angles(
            angles_series,
            baseline_frames=int(head_baseline_frames),
            yaw_delta_deg=float(head_yaw_delta_deg),
        )

        strategy = build_decision_strategy(mode=str(decision_mode), min_blinks=int(blink_min_count))
        signals = LivenessSignals(anti_spoof_passed=anti_passed, blink_count=int(blink_count), head_movement=bool(head_movement))
        live = decide_liveness(strategy=strategy, signals=signals)

        label = "REAL" if anti_passed else "FAKE"
        yaw_range = yaw_range_from_angles(angles_series, baseline_frames=int(head_baseline_frames))

        checks = LivenessChecks(
            blink_detected=bool(blink_count >= int(blink_min_count)),
            head_movement=bool(head_movement),
            blink_count=int(blink_count),
            anti_spoof=AntiSpoofCheck(label=label, confidence=float(conf_med), passed=bool(anti_passed)),
        )

        msg = (
            f"frames_with_face={frames_with_face} real_count={real_count} "
            f"conf_med={conf_med:.2f} blinks={blink_count} yaw_range={yaw_range:.1f}"
        )

        logger.info(msg)

        return LivenessResponse(
            liveness=bool(live),
            confidence=float(conf_med),
            checks=checks,
            message=msg,
        )


def _resize_max_side(frame: np.ndarray, *, max_side_px: int) -> np.ndarray:
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= int(max_side_px):
        return frame

    scale = float(max_side_px) / float(m)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(frame, (nw, nh))
