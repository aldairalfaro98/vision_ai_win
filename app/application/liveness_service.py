# app/application/liveness_service.py
import logging
from statistics import median
from typing import Dict, List, Optional
from app.domain.blink import detect_blink_from_ear_series
from app.domain.decision import decide_liveness
from app.domain.head_movement import detect_head_movement_from_angles

from app.domain.models import LivenessResponse, LivenessChecks
from app.domain.session_window import SessionWindow
from app.infrastructure.uniface.anti_spoof import (
    UniFaceAntiSpoofPredictor,
    decode_image_bytes_to_bgr,
)
from app.infrastructure.uniface.landmarks import UniFaceLandmarks
from app.infrastructure.uniface.pose import rvec_to_euler_degrees

logger = logging.getLogger("liveness")

_predictor = UniFaceAntiSpoofPredictor()
_landmarks = UniFaceLandmarks()
_sessions: Dict[str, SessionWindow] = {}



# LivenessService: servicio de verificación de vida
class LivenessService:

# Verifica liveness a partir de bytes de imagen
    def check(self, image_bytes: Optional[bytes]) -> LivenessResponse:
        if image_bytes is None:
            return self._empty()

        image_bgr = decode_image_bytes_to_bgr(image_bytes)
        anti = _predictor.predict_from_bgr(image_bgr)

        if anti is None:
            return self._empty()

        return LivenessResponse(
            liveness=anti.is_real,
            confidence=anti.confidence,
            checks=LivenessChecks(blink_detected=False, head_movement=False),
        )
    
# nuevo: streaming style (1 frame)
    def check_frame(self, *, image_bytes: bytes, session_id: str) -> LivenessResponse:
        win = self._get_window(session_id)
        sig = self._analyze_frame(image_bytes)
        if sig is None:
            live, conf, blink, head = win.compute()
            return self._resp(live, conf, blink, head)
        win.add(**sig)
        live, conf, blink, head = win.compute()
        logger.info(
            "frame_decision: sid=%s live=%s conf=%.3f blink=%s head=%s",
            session_id,
            live,
            conf,
            blink,
            head,
        )
        return self._resp(live, conf, blink, head)

    def _get_window(self, session_id: str) -> SessionWindow:
        if session_id not in _sessions:
            _sessions[session_id] = SessionWindow(window_size=8, min_real=3, threshold=0.6)
        return _sessions[session_id]

    def _analyze_frame(self, image_bytes: bytes):
        img = decode_image_bytes_to_bgr(image_bytes)
        anti = _predictor.predict_from_bgr(img)
        if anti is None:
            return None
        ear_avg, angles = self._signals_from_bbox(img, anti.bbox)
        return {
            "is_real": bool(anti.is_real),
            "confidence": float(anti.confidence),
            "ear_avg": ear_avg,
            "angles": angles,
        }

    def _signals_from_bbox(self, img, bbox):
        lm = _landmarks.analyze(img, bbox)
        if not lm:
            return None, None
        ear_avg = float((lm.ear_left + lm.ear_right) / 2.0)
        angles = rvec_to_euler_degrees(lm.rvec)
        return ear_avg, angles

    def _resp(self, live: bool, conf: float, blink: bool, head: bool) -> LivenessResponse:
        return LivenessResponse(
            liveness=bool(live),
            confidence=float(max(0.0, min(1.0, conf))),
            checks=LivenessChecks(blink_detected=bool(blink), head_movement=bool(head)),
        )

    def _empty(self) -> LivenessResponse:
        return self._resp(False, 0.0, False, False)

# Extra method for multiple frames

    def check_frames(
        self,
        frames_bytes: List[bytes],
        *,
        min_real: int = 3,
        threshold: float = 0.6,
    ) -> LivenessResponse:
        confidences: List[float] = []
        real_count = 0
        ear_series: List[float] = []
        pose_angles: List[tuple[float, float, float]] = []

        logger.info(
            "check_frames: received=%d min_real=%d threshold=%.2f",
            len(frames_bytes),
            min_real,
            threshold,
        )

        for i, b in enumerate(frames_bytes):
            try:
                img = decode_image_bytes_to_bgr(b)
            except Exception as e:
                logger.warning("frame[%d]: decode_failed err=%s", i, e)
                continue

            anti = _predictor.predict_from_bgr(img)
            if anti is None:
                logger.info("frame[%d]: no_face_detected", i)
                continue

            logger.info("frame[%d]: is_real=%s confidence=%.4f", i, anti.is_real, anti.confidence)

            confidences.append(anti.confidence)
            if anti.is_real:
                real_count += 1

           # ---- 2.5.1: Landmarks + logs (EAR + pose), reusando bbox ----
            try:
                lm = _landmarks.analyze(img, anti.bbox)
                if lm:
                    ear_avg = float((lm.ear_left + lm.ear_right) / 2.0)
                    ear_series.append(ear_avg)
                    angles = rvec_to_euler_degrees(lm.rvec)
                    pose_angles.append(angles)
                    logger.info("frame[%d]: ear_left=%.4f ear_right=%.4f", i, lm.ear_left, lm.ear_right)
                    logger.info("frame[%d]: angles(y,p,r)=%s", i, angles)
                    logger.info("frame[%d]: rvec=%s", i, lm.rvec.reshape(-1))
            except Exception as e:
                logger.warning("frame[%d]: landmarks_or_pose_failed err=%s", i, e)
            # -------------------------------------------------------------


        if not confidences:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        conf_med = float(median(confidences))
        anti_spoof_passed = (real_count >= min_real) and (conf_med >= threshold)
        blink_detected = detect_blink_from_ear_series(ear_series)
        head_movement = detect_head_movement_from_angles(pose_angles)
        liveness_final = decide_liveness(
            anti_spoof_passed=anti_spoof_passed,
            blink_detected=blink_detected,
            head_movement=head_movement,
        )
        logger.info(
            "decision: anti_spoof_passed=%s blink=%s head=%s => liveness=%s",
            anti_spoof_passed,
            blink_detected,
            head_movement,
            liveness_final,
        )
        #Log de ear_series y blink_detected
        logger.info("blink: ear_series_len=%d blink_detected=%s", len(ear_series), blink_detected)

        logger.info(
            "check_frames: processed=%d real_count=%d median_conf=%.4f liveness=%s",
            len(confidences),
            real_count,
            conf_med,
            liveness_final,
        )

        return LivenessResponse(
            liveness=liveness_final,
            confidence=conf_med,
            checks=LivenessChecks(blink_detected=blink_detected, head_movement=head_movement),
        )

