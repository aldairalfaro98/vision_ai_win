# app/application/liveness_service.py
from typing import Optional, List
from statistics import median

from app.domain.models import LivenessResponse, LivenessChecks
from app.infrastructure.uniface.anti_spoof import (
    UniFaceAntiSpoofPredictor,
    decode_image_bytes_to_bgr,
)

_predictor = UniFaceAntiSpoofPredictor()


# LivenessService: servicio de verificación de vida
class LivenessService:

# Verifica liveness a partir de bytes de imagen
    def check(self, image_bytes: Optional[bytes]) -> LivenessResponse:
        if image_bytes is None:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        image_bgr = decode_image_bytes_to_bgr(image_bytes)
        anti = _predictor.predict_from_bgr(image_bgr)

        if anti is None:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        return LivenessResponse(
            liveness=anti.is_real,
            confidence=anti.confidence,
            checks=LivenessChecks(blink_detected=False, head_movement=False),
        )
    
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

        for b in frames_bytes:
            try:
                img = decode_image_bytes_to_bgr(b)
            except Exception:
                continue

            anti = _predictor.predict_from_bgr(img)
            if anti is None:
                continue

            confidences.append(anti.confidence)
            if anti.is_real:
                real_count += 1

        if not confidences:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        conf_med = float(median(confidences))
        is_live = (real_count >= min_real) and (conf_med >= threshold)

        return LivenessResponse(
            liveness=is_live,
            confidence=conf_med,
            checks=LivenessChecks(blink_detected=False, head_movement=False),
        )

